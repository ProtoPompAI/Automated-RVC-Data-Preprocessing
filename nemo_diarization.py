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
from functools import reduce
import contextlib
import logging
logging.disable(logging.CRITICAL) # Hiding import warnings
# Supresses warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from omegaconf.errors import ConfigAttributeError, ConfigKeyError
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
# Supresses warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

def fmt_sec(col, round_to_zero=False, always_show_hour=True):
  def fmt_item(seconds):
    hours        = int(seconds // (60*60))
    minutes      = int(seconds % (60*60) // 60)
    seconds_disp = int(seconds % (60*60) % 60)
    miliseconds = round(seconds % 1, 2)

    base_fmt = f"{minutes:02d}:{seconds_disp:02d}"
    if (miliseconds != 0) and (not round_to_zero):
      base_fmt += str(miliseconds).split('.')[-1]
    if not (always_show_hour == False and hours == 0):
      base_fmt = f"{hours:02d}:" + base_fmt
    return base_fmt

  return(col.apply(lambda x: fmt_item(x)))

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
      f"ffmpeg -ss {row['Start']} -to {row['End'] + .2} "
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
  df_s = pd.DataFrame(result["segments"])
  df_s = df_s.assign(**{
      'Start_Formatted': fmt_sec(df_s['start'], round_to_zero=True, always_show_hour=False),
      'End_Formatted': fmt_sec(df_s['end'],  round_to_zero=True, always_show_hour=False)
    })  \
    .rename(columns={'start': 'Start', 'end': 'End', 'text': 'Text'}) \
    [['Start_Formatted', 'End_Formatted', 'Text', 'Start', 'End']]
  df_s.to_csv(out_file_path, index=False)

  DEBUG = True
  if DEBUG:
    temp_test_dir = Path(out_file_path).parents[0] / 'whisper_test'
    temp_test_dir.mkdir(exist_ok=True)

    reference_file, temp_test_dir = Path(audio_file_path), Path(temp_test_dir)
    
    for path in temp_test_dir.glob('*'):
      if path.is_dir():
        shutil.rmtree(path)
      else:
        path.unlink()

    for _, row in df_s.iterrows():
      fmt_num = lambda x: str(timedelta(0, round(x))).replace(':','-')
      out_file_name = f"'{fmt_num(row['Start'])}_to_{fmt_num(row['End'])}.wav'"
      subprocess.check_call(
        f"ffmpeg -ss {row['Start']} -to {row['End'] + .2} "
        f"-i {convert_pth(reference_file)} "
        f"{convert_pth(temp_test_dir / out_file_name)} -y",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
      )

  
  # 3. Creating VAD

  # https://github.com/NVIDIA/NeMo/discussions/6700#discussioncomment-5983801
  with open(out_file_path.with_suffix('.json'), 'w') as fp:
    for iloc in range(len(df_s)):
      row = df_s.iloc[iloc]
      vad_data = {
        "audio_filepath": convert_pth(audio_file_path),
        "offset": round(row['Start'], 2),
        "duration": round(row['End'] - row['Start'], 2),
        "label": "UNK",
        "uniq_id": audio_file_path.stem
      }
      json.dump(vad_data, fp)
      fp.write('\n')

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

def diarize(input_path, data_dir, domain_type='general', vad_path=None, nemo_asr=False, min_sec_cutoff_for_rttm=.1):
  """
  Creates a uses a NeMo Diarizer
  ::asr:: -> Uses the NeMO asr functionationaly.
    Tends to have lower accuracy with default options, so using it is not recommened.
    Instead, the whisperX library is used for a rough level sentence ASR in
    another function

  ::min_sec_cutoff_for_rttm:: Minimum second cutoff to keep a diarization.
    If this value is set small, very small clips will be fed into training models.  
  """ 

  use_external_vad = vad_path is not None

  data_dir.mkdir(exist_ok=True)

  cfg = OmegaConf.load(get_config(data_dir, domain_type))

  def safe_cfg_assign(key, value):
    """
    Function to check to see if a key exists in cfg before assigning.
    ::key:: Key to assign. Cannot assign over an existing dictionary for safety.
    ::value:: Value to assign. Cannot be a dictionary for safety.
    """

    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    def rsetattr(obj, attr, val):
      pre, _, post = attr.rpartition('.')
      return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
      def _getattr(obj, attr):
        return getattr(obj, attr, *args)
      return reduce(_getattr, [obj] + attr.split('.'))
    
    if type(value) in [dict, DictConfig]:
      raise KeyError(
        f'Value assigned would be a dictionary. '
        'Not allowed to assign dictionaries for safety. '
      )
      
    if key not in ['diarizer.manifest_filepath', 'diarizer.out_dir']:
      existing_val_check = rgetattr(cfg, key) # Check to see if specificed key exists in config
      if type(existing_val_check) in [dict, DictConfig]:
        raise KeyError(
          f'`{key}` is a refers to a dictionary in the config. '
          'Not allowed to assign over dictionaries for safety. '
          'Must specify each dictionary key individually.'
        )

    rsetattr(cfg, key, value)

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


  cfg_updates_base = {
    'diarizer.manifest_filepath': input_manifest_path,
    'diarizer.out_dir': convert_pth(data_dir), # Directory to store intermediate files and prediction outputs
    'diarizer.speaker_embeddings.model_path': 'titanet_large',
  }
  cfg_updates_hyper_param = {
    'sample_rate': 16_000,
    'batch_size': 64,
    
    'diarizer.clustering.parameters.embeddings_per_chunk': 100_000, # Optional parameter to tune. Default = 10_000
    'diarizer.clustering.parameters.chunk_cluster_count': 500, # Optional parameter to tune. Default = 50
    'diarizer.clustering.parameters.sparse_search_volume': 100,  # The higher the number, the more values will be examined with more time. Default = 10
    'diarizer.clustering.parameters.max_num_speakers': 50, # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored. Default = 8
    'diarizer.clustering.parameters.enhanced_count_thres': 80_000, # If the number of segments is lower than this number, enhanced speaker counting is activated. Default = 80
    'diarizer.clustering.parameters.enhanced_count_thres': .9, # Determines the range of p-value search: 0 < p <= max_rp_threshold. Default: .25

    "diarizer.speaker_embeddings.parameters.window_length_in_sec": [1.9,1.2,0.5], # Default [1.9,1.2,0.5]
    "diarizer.speaker_embeddings.parameters.shift_length_in_sec":  [0.95,0.6,0.25], # Default [0.95,0.6,0.25]
    "diarizer.speaker_embeddings.parameters.multiscale_weights":   [1,1,1], # Default [1,1,1]
    "diarizer.speaker_embeddings.parameters.save_embeddings": True,
    
    # 'diarizer.msdd_model.model_path': 'diar_msdd_telephonic', # Telephonic speaker diarization model 
    # 'diarizer.msdd_model.parameters.sigmoid_threshold': [0.7, 1.0], # Evaluate with T=0.7 and T=1.0
  }
  
  # Vad
  if use_external_vad:
    cfg_updates_hyper_param = {
      **cfg_updates_hyper_param,
      'diarizer.vad.model_path': None,
      'diarizer.vad.external_vad_manifest': str(convert_pth(vad_path)),
    }
  else: # Using External VAD
    cfg_updates_hyper_param = {
      **cfg_updates_hyper_param,
      'diarizer.vad.model_path': 'vad_multilingual_marblenet',
      'diarizer.vad.parameters.pad_offset': .2, # Adding durations after each speech segment. Default .2
      'diarizer.vad.parameters.pad_onset': .25, # Adding durations before each speech segment. Default .2
      'diarizer.vad.parameters.window_length_in_sec': .63, # Window length in sec for VAD context input. Default .63
      'diarizer.vad.parameters.shift_length_in_sec': .08 , # Shift length in sec for generate frame level VAD prediction. Default .08 
    }

  if nemo_asr:
    arpa_model_path = str(resolve_cache_dir() / '4gram_big.arpa')
    if not Path(arpa_model_path).exists():
      dl_arpa_model()
    
    cfg_updates_hyper_param = {
      **cfg_updates_hyper_param,
      "diarizer.oracle_vad": False, # ----> Not using oracle VAD 
      "diarizer.asr.parameters.asr_based_vad": False,
      "diarizer.asr.model_path": 'stt_en_conformer_ctc_large',
      "diarizer.asr.ctc_decoder_parameters.pretrained_language_model": arpa_model_path,
      "diarizer.asr.realigning_lm_parameters.arpa_language_model": arpa_model_path,
      "diarizer.asr.realigning_lm_parameters.logprob_diff_threshold": 1.2,
    }
    for key, item in {**cfg_updates_base, **cfg_updates_hyper_param}.items():
      safe_cfg_assign(key, item)

    asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()
    word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
  
    with NoStdStreams():
      diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

    with open(data_dir / 'nemo_transcription_info.json', 'w') as f:
      json.dump(trans_info_dict, f)

  else:
    for key, item in {**cfg_updates_base, **cfg_updates_hyper_param}.items():
      safe_cfg_assign(key, item)
    
    diarizer_model = ClusteringDiarizer(cfg=cfg) # diarizer_model._cfg
    # print(diarizer_model._cfg)
    with NoStdStreams(): 
      diarizer_model.diarize()

  df_rttm = pd.read_csv(
    data_dir / f'pred_rttms/{input_path.stem}.rttm',
    delimiter='\s', header=None,
    usecols=[5,8,11], names=['Start','Clip_Length','Speaker'],  
    engine='python'
  )
  df_rttm = df_rttm \
    .assign(**{
      'End': df_rttm['Start'] + df_rttm['Clip_Length'],
      'Clip_Over_Min_Secs': (df_rttm['Clip_Length'] > min_sec_cutoff_for_rttm),
      'Start_Formatted': fmt_sec(df_rttm['Start']),
      'End_Formatted': fmt_sec(df_rttm['Start'] + df_rttm['Clip_Length'])
    })
  df_rttm.to_csv(data_dir / 'Diarization_Formatted.csv', index=False)
  add_transcription_to_diarization(data_dir / 'Diarization_Formatted.csv', data_dir / 'whisperx_transcription.csv')

def diarize_clip_transcribe(input_path, output_directory, clip_audio):
  output_directory = Path(output_directory)
  output_directory.mkdir(exist_ok=True)
  
  if clip_audio:
    (output_directory / 'diarization'  ).mkdir(exist_ok=True)
    (output_directory / 'clipped_audio').mkdir(exist_ok=True)
    dia_path = output_directory / 'diarization'
  else:
    dia_path = output_directory
  
  generate_whisperx_transcription(input_path, dia_path / 'whisperx_transcription.csv')
  diarize(input_path, dia_path, vad_path=(dia_path / 'whisperx_transcription.json'))
  if clip_audio:
    clip_from_rttm(dia_path / 'Diarization_Formatted.csv', input_path, (output_directory / 'clipped_audio'))
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path', help='Input path. Needs to a mono .wav file or a directory of mono .wav files')
  parser.add_argument('output_directory', help='Output directory')
  parser.add_argument('-c', '--clip_audio', action='store_true', help='Use generated rttm file to clip audio')
  args = parser.parse_args()
  
  input_path = Path(args.input_path)
  output_directory = Path(args.output_directory)

  if input_path.is_dir():
    input_files = list(input_path.glob('*.wav'))
    progress = tqdm(desc='Diarizing individual files', total=len(input_files))

    for idx in range(len(input_files)):
      input_file = input_files[idx]
      diarize_clip_transcribe(input_file, output_directory / input_file.stem, args.clip_audio)
      progress.update(1)
  else:
    diarize_clip_transcribe(input_path, output_directory, args.clip_audio)
