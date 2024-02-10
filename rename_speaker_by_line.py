import shutil, os, subprocess, argparse, mimetypes, re
from datetime import datetime
from functools import partial
from pathlib import Path
import spacy
import pandas as pd

convert_pth = lambda x: f'"{str(Path(x).absolute().as_posix())}"'
s_call = partial(subprocess.check_call, shell=True, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL)

def split_by_min(out_folder, split_minutes=5):

  # Function that is called to cut the audio by timestamps if required
  def cut_to_timestamp(start_time, end_time, in_path, out_path):
    s_call(
      f"ffmpeg -i {convert_pth(in_path)} -ss {start_time} -to {end_time} "
      f"-c:v copy -c:a copy {convert_pth(out_path)} -y", shell=True
    )

  # Convert python timestamp to seconds
  def conv_to_sec(x):
    return (x.hour * 60 * 60) + (x.minute * 60) + x.second + float(f".{x.microsecond}")

  # Uses FFMPEG to get the duration of a media file
  def get_clip_len(file_path):
    clip_len = str(subprocess.check_output(
      f'ffmpeg -i {convert_pth(file_path)} 2>&1 | grep "Duration"', shell=True
    ))
    clip_len = re.search(r'Duration: ([0-9:\.]*),', clip_len).group(1)
    clip_len = datetime.strptime(clip_len, '%H:%M:%S.%f')
    clip_secs = conv_to_sec(clip_len)
    return clip_len, clip_secs

  def fmt_sec(seconds, file_name_style=False):
    hours        = int(seconds // (60*60))
    minutes      = int(seconds % (60*60) // 60)
    seconds_disp = int(seconds % (60*60) % 60)
    milliseconds = seconds % 1
    if file_name_style == False:
      if milliseconds != 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_disp:02d}.{str(milliseconds).split('.')[-1]}"
      else:
        return f"{hours:02d}:{minutes:02d}:{seconds_disp:02d}"
    else:
      return f"{hours:02d}-{minutes:02d}"

  in_audio_file = list(out_folder.glob('*'))[0]

  clip_len, clip_secs = get_clip_len(in_audio_file)
  for sec_start in range(0, int(clip_secs) + 1, split_minutes * 60):
    sec_end = min(sec_start + split_minutes * 60, clip_secs)
    cut_to_timestamp(
      fmt_sec(sec_start), fmt_sec(sec_end), in_audio_file,
      in_audio_file.parent / (f"{fmt_sec(sec_start, True)}_to_" +
                  f"{fmt_sec(sec_end, True)}.wav")
    )
  
  in_audio_file.unlink()

def combine_wav(out_folder):
    def get_media_paths(in_path):
      """
      Checks for audio paths in a given directory
      """
      media_paths = []
      for path in Path(in_path).glob('*'):
        if path.is_dir():
          continue
        file_type = mimetypes.guess_type(path)[0]
        if file_type != None:
          if file_type.split('/')[0] in ['audio', 'video']:
            media_paths.append(path)
      if media_paths == []:
        raise ValueError(f'No media files found in input folder: {in_path}.')
      return media_paths

    media_paths = get_media_paths(out_folder)

    # Making a file with all of the audio files for ffmpeg 
    with open(out_folder / 'Files_to_combine.txt', 'w') as text_file:
      for wav_file in media_paths:
        wav_file = str(convert_pth(wav_file))
        wav_file = wav_file.replace('"',"'")
        text_file.write(f"file {wav_file}")
        text_file.write('\n')

    # Using FFMPEG to combine all of the wav files
    s_call(
      f"ffmpeg -f concat -safe 0 -i "
      f"{convert_pth(out_folder / 'Files_to_combine.txt')} " 
      f"-c copy {convert_pth(out_folder / 'audio.wav')} -y"
    )

    for path in out_folder.glob('*'):
      if path == out_folder / 'audio.wav':
        continue
      if not path.is_dir():
        path.unlink()
    
def load_diarization():
  """
  Loads diarization in csv format
  """
  dtypes = {
    str: ['File_Name', 'Speaker', 'Start_Formatted', 'End_Formatted',
          'Highest_Likelihood_Line', 'In_Audio_Path'],
    float: ['Start', 'Clip_Length', 'End'],
    bool: ['Clip_Over_Min_Secs']
  }

  return pd.read_csv(
    Path(args.input_diarization_directory) / 'Diarization_Formatted.csv',
    dtype={v: k for k in dtypes.keys() for v in dtypes[k]}
  )

def reset_speaker_name(folder_name, check_name, dia_path):
  """
  Renaming previously changed folder to default
  Required if going over the data twice  
  """
  def rename_speaker(row, i):
    if str(row['Speaker']) == str(check_name) and str(row['File_Name']) == str(folder_name.name):
      return f'speaker_{i}'
    else:
      return row['Speaker']

  for i in range(0, 10_000):
    speaker_check_folder = (folder_name / f'speaker_{i}')
    if not speaker_check_folder.exists():
      (folder_name / check_name).rename(speaker_check_folder)
      assert not (folder_name / check_name).exists()
      df_t = pd.read_csv(dia_path)
      df_t['Speaker'] = df_t.apply(lambda x: rename_speaker(x, i), axis=1)
      assert str(check_name) not in df_t[df_t['File_Name'].astype(str) == str(folder_name.name)]['Speaker'].unique()
      df_t.to_csv(dia_path, index=False)
      return
  raise ValueError()

def rename_by_line(spec_series, rename_folder_to,
                   diarization_csv, clipped_audio, file_name, skip_user_verification=False):
  """
  Renames the speaker by a specified line.
  Uses spacy to provide a truthy match to the whipserX transcription.
  """
  # https://stackoverflow.com/questions/55921104/spacy-similarity-warning-evaluating-doc-similarity-based-on-empty-vectors
  os.environ["SPACY_WARNING_IGNORE"] = "W008"
  line_to_check_for = spec_series['Line']
  try:
    nlp = spacy.load("en_core_web_lg")
  except OSError:
    subprocess.call('python -m spacy download en_core_web_lg', shell=True)
    nlp = spacy.load("en_core_web_lg")

  diarization_csv = Path(diarization_csv, dtype={'File_Name': str})
  clipped_audio = Path(clipped_audio)

  df = load_diarization()
  if file_name not in df['File_Name'].unique():
    raise ValueError(f'Specified File Name: {file_name} not found in diarization file.')
  df = df[df['File_Name']==file_name]
  df = df \
    .assign(**{
      'Similarity_Score': df['Highest_Likelihood_Line'].apply(
        lambda x: 0 if pd.isna(x) or x.strip() == ''
        else nlp(x).similarity(nlp(line_to_check_for))
      )
    }) \
    .reset_index() \
    .sort_values('Similarity_Score', ascending=False)

  if len(df['Speaker'].unique()) == 1:
    if str(spec_series['Finalize']).lower() in ['yes', 'true']:
      print(f'Warning, only a single speaker found for folder: `{clipped_audio.name}`.')

  if len(df[df['Similarity_Score'] >= .75]) == 0:
    print(
      f'Could not find line {line_to_check_for}. Highest likelihood lines:'
      f"{df[['Highest_Likelihood_Line', 'Similarity_Score', 'Speaker']].head(3)}",
      sep='\n'
    )
  else:
    best_fit_row = df.iloc[0]
    if skip_user_verification != True:
      resp = input(f'Use line: {best_fit_row["Highest_Likelihood_Line"]}? Y/n\n')
      if resp.lower() not in ['y', 'yes']:
        print('Yes (Y/yes) not detected, exiting.')
        exit()
    print(f'Changing folder name `{clipped_audio.name}/{best_fit_row["Speaker"]}` to `{clipped_audio.name}/{rename_folder_to}`')
    Path(clipped_audio / best_fit_row['Speaker']).rename(clipped_audio / rename_folder_to) 

    df_out = df.copy() \
      .assign(**{
        'Speaker': df['Speaker'].replace(best_fit_row['Speaker'], rename_folder_to)
      }) \
      .sort_values('index') \
      .drop(columns=['Similarity_Score', 'index']) \
    
    if file_name is not None:
      df_append = load_diarization()
      df_append = df_append[df_append['File_Name'] != file_name]
      df_out = pd.concat([df_out, df_append])
      df_out = df_out.sort_values(['File_Name', 'Speaker', 'Start_Formatted',	'End_Formatted'])

    df_out.to_csv(diarization_csv, index=False)

def rename_from_spec_file(base_folder, check_name, spec_file, move_results_to_folder):
  to_check = pd.read_excel(
    spec_file, dtype={'Speaker':str,	'Episode':str, 'Line':str, 'Finalize': str,}
  )
  if not check_name in to_check['Speaker'].unique():
    raise ValueError(
      f'Error. Speaker {check_name} not found in speakers: {to_check["Speaker"].unique()}')
  to_check = to_check[to_check['Speaker'] == check_name]
  dia_path = base_folder / 'Diarization_Formatted.csv'

  # For loop for each filtered row of the to_check file.
  for idx in range(len(to_check)):
    row = to_check.iloc[idx]
    folder_name = Path(base_folder / f'clipped_audio/{row["Episode"]}')
    if not folder_name.exists():
      # raise ValueError(f'{folder_name} not found')
      print(f'`{folder_name}` from specified lines not found. Skipping.')
      continue

    df_test = load_diarization()
    df_test = df_test[
      (df_test['File_Name'] == folder_name.name) &
      (df_test['Speaker'] == check_name)
    ]
    reset_cond = (
      (Path(folder_name) / check_name).exists() or
      len(df_test) != 0
    )
    if reset_cond:
      reset_speaker_name(folder_name, check_name, dia_path)
    
    rename_by_line(
      row, check_name, diarization_csv=dia_path,
      clipped_audio=folder_name, file_name=folder_name.name, skip_user_verification=True
    )

    if str(row['Finalize']).lower() in ['yes', 'true']:
      for path in (folder_name / check_name).glob('*.wav'):
        shutil.copy(path, move_results_to_folder / (folder_name.name+'_'+path.name))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_diarization_directory', help='The result directory from nemo_diarization.py. Used as input.')
  parser.add_argument('specification_file', help='Excel file that specifies the speakers through a line.')
  parser.add_argument('speaker_name', help='Speaker label. Needs to be present in the Excel specification file.')
  parser.add_argument('results_directory', help='Will copy clips with the given speaker_name to this directory if finalize '
                      'is true in the specification_file.')
  parser.add_argument('-seg', '--segment', required=False, help='Break the data up into SEGMENT minutes')
  args = parser.parse_args()

  input_diarization_directory = Path(args.input_diarization_directory)
  specification_file = Path(args.specification_file)
  speaker_name = args.speaker_name
  results_directory = Path(args.results_directory)
  results_directory.mkdir(exist_ok=True)
  for path in results_directory.glob('*'):
    path.unlink()

  rename_from_spec_file(input_diarization_directory, speaker_name, specification_file, results_directory)
  if args.segment:
    combine_wav(results_directory)
    split_by_min(results_directory, split_minutes=args.segment)
