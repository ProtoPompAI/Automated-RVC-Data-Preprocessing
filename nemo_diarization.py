# Notes for code implementation
# https://lajavaness.medium.com/comparing-state-of-the-art-speaker-diarization-frameworks-pyannote-vs-nemo-31a191c6300
# https://github.com/NVIDIA/NeMo/issues/5174
# https://github.com/NVIDIA/NeMo/pull/7737#issuecomment-1808724046
# https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb#scrollTo=CwtVUgVNBR_P
# https://lajavaness.medium.com/deep-dive-into-nemo-how-to-efficiently-tune-a-speaker-diarization-pipeline-d6de291302bf
# https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_with_asr_infer.py



import json, shutil, subprocess, argparse, gzip, os
from datetime import timedelta
from pathlib import Path
import contextlib
import logging
logging.disable(logging.CRITICAL) # Hiding import warnings

from omegaconf import OmegaConf
import wget
import pandas as pd
import whisperx

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.utils.data_utils import resolve_cache_dir 
import sys
from tqdm import tqdm

logging.disable(logging.NOTSET) # Getting logger back




class NoStdStreams(object):
  """
  Supresses TQDM output when used as context
  """
  # https://github.com/NVIDIA/NeMo/discussions/3281
  def __init__(self, stdout = None, stderr = None):
    self.devnull = open(os.devnull,'w')
    self._stdout = stdout or self.devnull or sys.stdout
    self._stderr = stderr or self.devnull or sys.stderr

  def __enter__(self):
    self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
    self.old_stdout.flush(); self.old_stderr.flush()
    sys.stdout, sys.stderr = self._stdout, self._stderr

  def __exit__(self, exc_type, exc_value, traceback):
    self._stdout.flush(); self._stderr.flush()
    sys.stdout = self.old_stdout
    sys.stderr = self.old_stderr
    self.devnull.close() 


# Supresses warnings
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# logging.getLogger("pytorch").setLevel(logging.ERROR)
# logging.getLogger("nemo_logger").setLevel(logging.ERROR)
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

# loggers

convert_pth = lambda x: Path(x).absolute().as_posix()

def dl_arpa_model():
  def gunzip(file_path,output_path):
      with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
          shutil.copyfileobj(f_in, f_out)
          f_in.close()
          f_out.close()
  ARPA_URL = 'https://kaldi-asr.org/models/5/4gram_big.arpa.gz'

  if not (Path(resolve_cache_dir()) / '4gram_big.arpa').exists():
    if not (Path(resolve_cache_dir()) / '4gram_big.arpa.gz').exists():
      f = wget.download(ARPA_URL, str(resolve_cache_dir))
    else:
      f = str(Path(resolve_cache_dir()) / '4gram_big.arpa.gz')
    gunzip(f,f.replace(".gz",""))
    Path(f).unlink(True)

def get_config(data_dir, domain_type='general'):
  assert domain_type in ['general', 'meeting', 'telephonic']
  config_file_name =  f"diar_infer_{domain_type}.yaml"
  if not os.path.exists(data_dir / config_file_name):
    config = wget.download(
      f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{config_file_name}",
      str(data_dir.absolute().as_posix())
    )
  else:
    config = data_dir / config_file_name
  return Path(config).absolute().as_posix()

def clip_from_rttm(rttm_path, reference_file, out_folder):
  """
  Uses RTTM file from diarization to create an audio folder for each speaker
  Utilizes the Pandas library + ffmpeg
  """

  reference_file, out_folder = Path(reference_file), Path(out_folder)
  df_rtm = pd.read_csv(rttm_path)
  df_rtm = df_rtm[df_rtm['Clip_Over_Min_Secs'] == True]

  for folder in out_folder.glob('*'):
    shutil.rmtree(folder)

  for unique_speaker in df_rtm['Speaker'].unique():
    # if os.path.exists(out_folder / unique_speaker):
    #   shutil.rmtree(out_folder / unique_speaker)
    (out_folder / unique_speaker).mkdir(exist_ok=True)

  for _, row in df_rtm.iterrows():
    fmt_num = lambda x: str(timedelta(0, round(x))).replace(':','-')
    out_file_name = f"'{fmt_num(row['Start'])}_to_{fmt_num(row['End'])}.wav'"
    subprocess.check_call(
      f"ffmpeg -ss {row['Start']} -to {row['End']} "
      f"-i {convert_pth(reference_file)} "
      f"{convert_pth(out_folder / row['Speaker'] / out_file_name)} -y",
      shell=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL
    )

def generate_whisperx_transcription(audio_file_path, out_file_path, batch_size=8, compute_type="float16",
                                   device="cuda", language='en'):
  """
  https://github.com/m-bain/whisperX#python-usage--

  ::audio_file_path:: path to audio file
  ::out_file_path:: path to output path
  ::batch_size:: batch size, reduce if low on GPU mem
  ::compute_type:: compute type, change to "int8" if low on GPU mem (may reduce accuracy)
  ::device:: device for inference, GPU significantly decreases inference time
  ::language:: specified language will be decrease inference time
  """

  # 1. Transcribe with original whisper (batched)
  with contextlib.redirect_stdout(None):
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)

  # 2. Align whisper output
  with contextlib.redirect_stdout(None):
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
  df_s = pd.DataFrame(result["segments"]) \
    [['start', 'end',	'text']] \
    .rename(columns={'start': 'Start', 'end': 'End', 'text': 'Text'})
  df_s.to_csv(out_file_path, index=False)

def add_transcription_to_diarization(dia_path, trans_path):
  """
  Uses the transciption from whisperX to add a highest likelihood line to 
  the NeMo diarization.
  While NeMo does have transcription capabilites, whisperX typically has 
  very accurate results.

  Overwrites the dia_path provided with the new .csv file
  """
  
  df_d = pd.read_csv(dia_path)
  df_t = pd.read_csv(trans_path)

  # Assigning the highest chance line to the RTM File
  # Based on the closeness of Start + End.
  df_d['Highest_Likelihood_Line'] = None
  for idx in df_d.index:
    if df_d.loc[idx]['Clip_Over_Min_Secs'] == False:
      df_d.at[idx, 'Highest_Likelihood_Line'] = None
    df_d[df_d['Clip_Over_Min_Secs']==True]
    
    time_alignment = (
      abs(df_d.loc[idx]['Start'] - df_t['Start']) +
      abs(df_d.loc[idx]['End'] - df_t['End'])
    )
    df_d.at[idx, 'Highest_Likelihood_Line'] = \
      df_t.loc[time_alignment.argmin()]['Text']

  # This operation below ensures that there are no duplicate 
  # values for the column `Highest_Likelihood_Line` by
  # using highest `Clip_Length` for the distinct `Highest_Likelihood_Line`.
  # In the rare case of two `Highest_Likelihood_Line`'s having the same 
  # `Clip_Length`, use the first occurrence in the dataframe (also will be the
  # minimum `Start`) 

  df_g = df_d \
    .reset_index() \
    .groupby('Highest_Likelihood_Line') \
      .agg({
        'Clip_Length': 'max',
        'index': lambda x: min(list(x))
      }) \
    .reset_index() \
    .drop(columns=['Clip_Length']) \
    .set_index('index') \
    .squeeze() \
    .to_dict()
  df_d['Highest_Likelihood_Line'] = df_d.index.map(df_g)

  df_d.to_csv(dia_path, index=False)

def diarize(input_path, data_dir, domain_type='general', nemo_asr=False, min_sec_cutoff_for_rttm=.75):
  """
  Creates a uses a NeMo Diarizer
  ::asr:: -> Uses the NeMO asr functionationaly.
    Tends to have lower accuracy with default options, so using it is not recommened.
    Instead, the whisperX library is used for a rough level sentence ASR in
    another function

  ::min_sec_cutoff_for_rttm:: Minimum second cutoff to keep a diarization.
    If this value is set small, very small clips will be fed into training models.  
  """ 

  data_dir.mkdir(exist_ok=True)

  cfg = OmegaConf.load(get_config(data_dir, domain_type))
  input_manifest_path = str(convert_pth(data_dir / 'input_manifest.json'))
  with open(input_manifest_path, 'w') as fp:
    meta = {
      'audio_filepath': str(convert_pth(input_path)),
      'offset': 0, 
      'duration':None, 
      'label': 'infer', 
      'text': '-', 
      'num_speakers' : None, 
      'rttm_filepath': None, # You can add a reference file here 
      'uem_filepath' : None,
      'verbose': False,
    }
    json.dump(meta, fp)
    fp.write('\n')

  cfg.diarizer.manifest_filepath = input_manifest_path
  cfg.diarizer.out_dir = convert_pth(data_dir) # Directory to store intermediate files and prediction outputs

  cfg.sample_rate = 16_000
  cfg.batch_size = 64 # 64

  # Parameters that can be adjusted
  cfg.diarizer.speaker_embeddings.model_path = 'titanet_large'
  cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'

  cfg.diarizer.clustering.parameters.embeddings_per_chunk = 10_000 # Optional parameter to tune. Default = 10_000
  cfg.diarizer.clustering.parameters.chunk_cluster_count = 50 # Optional parameter to tune. Default = 50
  # cfg.diarizer.vad.parameters.window_length_in_sec = .8
  # cfg.diarizer.vad.parameters.shift_length_in_sec = .04
  cfg.diarizer.vad.parameters.pad_onset = .25    #  # Adding durations before each speech segment. Default .2
  cfg.diarizer.vad.parameters.pad_offset = .2 # Adding durations after each speech segment. Default .2
  cfg.verbose = False

  # cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.9,1.2,1.0,.75,0.5] # Default [1.9,1.2,0.5]
  # cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec =  [0.95,0.6,0.5,0.375,0.25] # Default  [0.95,0.6,0.25]
  # cfg.diarizer.speaker_embeddings.parameters.multiscale_weights =   [1   ,1    ,.8  ,  .8  ,1   ] 
  # cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False



  if nemo_asr:
    arpa_model_path = str(resolve_cache_dir() / '4gram_big.arpa')
    if not Path(arpa_model_path).exists():
      dl_arpa_model()

    cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
    cfg.diarizer.asr.parameters.asr_based_vad = False
    cfg.diarizer.asr.model_path='stt_en_conformer_ctc_large'
    cfg.diarizer.asr.ctc_decoder_parameters.pretrained_language_model = arpa_model_path
    cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model = arpa_model_path
    cfg.diarizer.asr.realigning_lm_parameters.logprob_diff_threshold = 1.2

    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()
    word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
  
    with contextlib.redirect_stdout(None):
      diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

    with open(data_dir / 'nemo_transcription_info.json', 'w') as f:
      json.dump(trans_info_dict, f)

  else:
    # import sys
    # old_stdout = sys.stdout # backup current stdout
    # sys.stdout = open(os.devnull, "w")
    
    diarizer_model = ClusteringDiarizer(cfg=cfg) # , verbose=False)
    # with contextlib.redirect_stderr(None):
    #   with contextlib.redirect_stdout(None):
    # diarizer_model.verbose = False
    with NoStdStreams(): 
      diarizer_model.diarize()

    # sys.stdout = old_stdout # reset old stdout

  df_rttm = pd.read_csv(
    data_dir / f'pred_rttms/{input_path.stem}.rttm',
    delimiter='\s', header=None,
    usecols=[5,8,11], names=['Start','Clip_Length','Speaker'],  
    engine='python'
  )
  df_rttm = df_rttm \
    .assign(**{
      'End': df_rttm['Start'] + df_rttm['Clip_Length'],
      'Clip_Over_Min_Secs': df_rttm['Clip_Length'] > min_sec_cutoff_for_rttm,
      'Start_Formatted': fmt_sec(df_rttm['Start']),
      'End_Formatted': fmt_sec(df_rttm['Start'] + df_rttm['Clip_Length'])
    })
  df_rttm.to_csv(data_dir / 'Diarization_Formatted.csv', index=False)


def diarize_clip_transcribe(input_path, output_directory, clip_audio, skip_transcription):
  output_directory = Path(output_directory)
  output_directory.mkdir(exist_ok=True)

  if clip_audio:
    (output_directory / 'diarization'  ).mkdir(exist_ok=True)
    (output_directory / 'clipped_audio').mkdir(exist_ok=True)
    dia_path = output_directory / 'diarization'
  else:
    dia_path = output_directory

  diarize(input_path, dia_path)
  if clip_audio:
    clip_from_rttm(dia_path / 'Diarization_Formatted.csv', input_path, (output_directory / 'clipped_audio'))
  if skip_transcription == False:
    generate_whisperx_transcription(input_path, dia_path / 'whisperx_transcription.csv')
    add_transcription_to_diarization(dia_path / 'Diarization_Formatted.csv', dia_path / 'whisperx_transcription.csv')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path', help='Input path. Needs to a mono .wav file or a directory of mono .wav files')
  parser.add_argument('output_directory', help='Output directory')
  parser.add_argument('-c', '--clip_audio', action='store_true', help='Use generated rttm file to clip audio')
  parser.add_argument('-s', '--skip_transcription', action='store_true', help='Skip the Whisper transcription process')
  args = parser.parse_args()
  
  input_path = Path(args.input_path)
  output_directory = Path(args.output_directory)

  if input_path.is_dir():
    input_files = list(input_path.glob('*.wav'))
    progress = tqdm(desc='Diarizing individual files', total=len(input_files))

    for idx in range(len(input_files)):
      input_file = input_files[idx]
      diarize_clip_transcribe(
        input_file, output_directory / input_file.stem,
        args.clip_audio, args.skip_transcription
      )
      progress.update(1)
  else:
    diarize_clip_transcribe(input_path, output_directory, args.clip_audio, args.skip_transcription)
