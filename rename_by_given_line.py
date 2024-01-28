# https://www.softwareok.com/?page=Windows/10/Quick-Tip/39
# Enter "extension:*.wav" 

# python nemo_diarization.py fmt_and_dia_trial/audio fmt_and_dia_trial/results -c && python rename_by_given_line.py "fmt_and_dia_trial/results/[bonkai77].Neon.Genesis.Evangelion.Episode.08.[BD.1080p.Dual-Audio.x265.HEVC.10bit]" "You're coming with me." "Asuka" -y


# import whisperx
import gc 
import pandas as pd
import os

# Uses a line to quickly pull a voice out.
import subprocess, argparse
from functools import partial
from pathlib import Path
import pandas as pd
import spacy

# /home/john/projects/utility/nemo_in_action/rename_by_given_line.py:36: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.
def rename_by_line(line_to_check_for, rename_folder_to, y, input_path_folder=None, diarization_csv=None, clipped_audio=None, file_name=None):
  os.environ["SPACY_WARNING_IGNORE"] = "W008" # https://stackoverflow.com/questions/55921104/spacy-similarity-warning-evaluating-doc-similarity-based-on-empty-vectors
  nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!
  if (input_path_folder is None) and (diarization_csv is None or clipped_audio is None):
    raise NotImplementedError()
  if input_path_folder:
    INP_L  = Path(input_path_folder) / 'diarization/Diarization_Formatted.csv'
    INP_SO = Path(input_path_folder) / 'clipped_audio'
  else:
    INP_L = Path(diarization_csv)
    INP_SO = Path(clipped_audio)

  SL = nlp(line_to_check_for)
  RN_TO = rename_folder_to
  df = pd.read_csv(INP_L)
  if file_name is not None:
    if file_name not in df['File_Name'].unique():
      raise ValueError(f'Specifed File Name: {file_name} not found in diarization file.')
    df = df[df['File_Name']==file_name]
  df = df \
    .assign(**{
      'Similarity_Score': df['Highest_Likelihood_Line'].apply(
        lambda x: 0 if pd.isna(x) or x.strip() == ''
        else nlp(x).similarity(SL)
      )
    }) \
    .reset_index() \
    .sort_values('Similarity_Score', ascending=False)

  if len(df['Speaker'].unique()) == 1:
    print(f'Warning, only a single speaker for file {Path(input_path_folder).name} found.')

  if len(df[df['Similarity_Score'] >= .75]) == 0:
    print(
      f'Could not find line {SL}. Highest likelihood lines:'
      f"{df[['Highest_Likelihood_Line', 'Similarity_Score', 'Speaker']].head(3)}",
      sep='\n'
    )
  else:
    best_fit_row = df.iloc[0]
    if not y:
      resp = input(f'Use line: {best_fit_row["Highest_Likelihood_Line"]}? Y/n\n')
      if resp.lower() not in ['y', 'yes']:
        print('Yes (Y/yes) not detected, exiting.')
        exit()
    print(f'Changing folder name `{best_fit_row["Speaker"]}` to `{RN_TO}`')
    Path(INP_SO / best_fit_row['Speaker']).rename(INP_SO / RN_TO) 

    df_out = df.copy() \
      .assign(**{
        'Speaker': df['Speaker'].replace(best_fit_row['Speaker'], RN_TO)
      }) \
      .sort_values('index') \
      .drop(columns=['Similarity_Score', 'index']) \
    
    if file_name is not None:
      df_append = pd.read_csv(INP_L)
      df_append = df_append[df_append['File_Name']!=file_name]
      df_out = pd.concat([df_out, df_append])
      df_out = df_out.sort_values(['File_Name', 'Speaker', 'Start_Formatted',	'End_Formatted'])

    df_out.to_csv(INP_L, index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path_folder' , help='Input path')
  parser.add_argument('line_to_check_for', help='Line to check for')
  parser.add_argument('rename_folder_to', help='Rename folder to')
  parser.add_argument('-y', '--yes', action='store_true',
                      help='Optional argument to auto-accept the top matching result', required=False)
  args = parser.parse_args()
  rename_by_line(args.line_to_check_for, args.rename_folder_to, args.yes, args.input_path_folder)