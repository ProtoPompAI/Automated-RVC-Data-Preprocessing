# https://www.softwareok.com/?page=Windows/10/Quick-Tip/39
# Enter "extension:*.wav" 

# python nemo_diarization.py fmt_and_dia_trial/audio fmt_and_dia_trial/results -c && python rename_by_given_line.py "fmt_and_dia_trial/results/[bonkai77].Neon.Genesis.Evangelion.Episode.08.[BD.1080p.Dual-Audio.x265.HEVC.10bit]" "You're coming with me." "Asuka" -y


# import whisperx
import gc 
import pandas as pd


# Uses a line to quickly pull a voice out.
import subprocess, argparse
from functools import partial
from pathlib import Path
import pandas as pd
import spacy

# /home/john/projects/utility/nemo_in_action/rename_by_given_line.py:36: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.

parser = argparse.ArgumentParser()
parser.add_argument('input_path_folder' , help='Input path')
# parser.add_argument('input_dir_speaker_outputs' , help='Input path', required=True)
# parser.add_argument('output_path', help='Output path')
parser.add_argument('line_to_check_for', help='Line to check for')
parser.add_argument('rename_folder_to', help='Rename folder to')
parser.add_argument('-y', '--yes', action='store_true', help='Optional argument to auto-accept the top matching result', required=False)

args = parser.parse_args()
nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!

# bp / 'diarization/Diarization_Formatted.csv'
bp = Path('/home/john/projects/utility/nemo_in_action/fmt_and_dia_trial/results/[bonkai77].Neon.Genesis.Evangelion.Episode.05.[BD.1080p.Dual-Audio.x265.HEVC.10bit].wav')

INP_L  = Path(args.input_path_folder) / 'diarization/Diarization_Formatted.csv'
INP_SO = Path(args.input_path_folder) / 'clipped_audio'
SL = nlp(args.line_to_check_for)
RN_TO = args.rename_folder_to

df = pd.read_csv(INP_L)
df = df \
  .assign(**{
    'Similarity_Score': df['Highest_Likelihood_Line'].apply(
      lambda x: 0 if pd.isna(x) else nlp(x).similarity(SL)
    )
  }) \
  .reset_index() \
  .sort_values('Similarity_Score', ascending=False)

if len(df[df['Similarity_Score'] >= .75]) == 0:
  print(
    f'Could not find line {SL}. Highest likelihood lines:'
    f"{df[['Highest_Likelihood_Line', 'Similarity_Score', 'Speaker']].head(3)}",
    sep='\n'
  )
else:
  best_fit_row = df.iloc[0]
  if not args.yes:
    resp = input(f'Use line: {best_fit_row["Highest_Likelihood_Line"]}? Y/n\n')
    if resp.lower() not in ['y', 'yes']:
      print('Yes (Y/yes) not detected, exiting.')
      exit()
  print(f'Changing folder name `{best_fit_row["Speaker"]}` to `{RN_TO}`')
  Path(INP_SO / best_fit_row['Speaker']).rename(INP_SO / RN_TO) 
  df \
    .assign(**{
      'Speaker': df['Speaker'].replace(best_fit_row['Speaker'], RN_TO)
    }) \
    .sort_values('index') \
    .drop(columns=['Similarity_Score', 'index']) \
    .to_csv(INP_L, index=False)

  
