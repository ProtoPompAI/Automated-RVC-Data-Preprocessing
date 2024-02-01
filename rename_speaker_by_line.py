import shutil, os, subprocess, argparse
from pathlib import Path
import spacy
import pandas as pd

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

def rename_by_line(line_to_check_for, rename_folder_to,
                   diarization_csv, clipped_audio, file_name, skip_user_verification=False):
  """
  Renames the speaker by a specified line.
  Uses spacy to provide a truthy match to the whipserX transcription.
  """
  # https://stackoverflow.com/questions/55921104/spacy-similarity-warning-evaluating-doc-similarity-based-on-empty-vectors
  os.environ["SPACY_WARNING_IGNORE"] = "W008"
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
    print(f'Warning, only a single speaker found.')

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
      row['Line'], check_name, diarization_csv=dia_path,
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
  args = parser.parse_args()

  input_diarization_directory = Path(args.input_diarization_directory)
  specification_file = Path(args.specification_file)
  speaker_name = args.speaker_name
  results_directory = Path(args.results_directory) 
  results_directory.mkdir(exist_ok=True)
  for path in results_directory.glob('*'):
    path.unlink()

  rename_from_spec_file(input_diarization_directory, speaker_name, specification_file, results_directory)
