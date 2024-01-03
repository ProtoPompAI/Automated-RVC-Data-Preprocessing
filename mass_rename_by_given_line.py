# https://www.softwareok.com/?page=Windows/10/Quick-Tip/39
# Enter "extension:*.wav" 

# python nemo_diarization.py fmt_and_dia_trial/audio fmt_and_dia_trial/results -c && python rename_by_given_line.py "fmt_and_dia_trial/results/[bonkai77].Neon.Genesis.Evangelion.Episode.08.[BD.1080p.Dual-Audio.x265.HEVC.10bit]" "You're coming with me." "Asuka" -y

# import whisperx
from pathlib import Path
import pandas as pd
from rename_by_given_line import rename_by_line
import argparse
# Uses a line to quickly pull a voice out.
import subprocess, argparse
from functools import partial
from pathlib import Path
import pandas as pd
import spacy
import shutil
import re

# /home/john/projects/utility/nemo_in_action/rename_by_given_line.py:36: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.

MOVE_TO_FOLDER = True
MOVE_TO_FOLDER_LOC = Path(
  '/home/john/projects/utility/nemo_in_action/fmt_and_dia_full_anime_trial_asuka'
)
CHECK_NAME = 'Asuka'

if __name__ == '__main__':
  if MOVE_TO_FOLDER:
    MOVE_TO_FOLDER_LOC.mkdir(exist_ok=True)
    for path in MOVE_TO_FOLDER_LOC.glob('*'):
      path.unlink()
  
  
  to_check = pd.read_excel('Check_For.xlsx', skiprows=3)
  
  for idx in range(len(to_check)):
    row = to_check.iloc[idx]
    folder_name = Path(
      'fmt_and_dia_full_anime_trial/results/['
      f'bonkai77].Neon.Genesis.Evangelion.Episode.'
      f'{row["Episode"]:02}.'
      '[BD.1080p.Dual-Audio.x265.HEVC.10bit]/'
    )
    if not folder_name.exists():
      folder_name = Path(
      'fmt_and_dia_full_anime_trial/results/['
      f'bonkai77].Neon.Genesis.Evangelion.Episode.'
      f'{row["Episode"]:02}.DC.'
      '[BD.1080p.Dual-Audio.x265.HEVC.10bit]/'
    )
    if not folder_name.exists():
      raise ValueError(folder_name)

    # Renaming named folder if already exists
    # Good for second pass  
    chk_f = (folder_name / 'clipped_audio')
    if (chk_f / CHECK_NAME).exists():
      dia_path = folder_name / 'diarization/Diarization_Formatted.csv'
      for i in range(0, 10_000):
        speaker_check = f'speaker_{i}'
        if not (chk_f / speaker_check).exists():
          (chk_f / CHECK_NAME).rename(chk_f / speaker_check)
          df_t = pd.read_csv(dia_path)
          df_t['Speaker'].replace(CHECK_NAME, f'speaker_{i}', inplace=True)
          df_t.to_csv(dia_path, index=False)
          break
 
    rename_by_line(
      row['Asuka Line'], CHECK_NAME, y=True,
      input_path_folder=str(folder_name.absolute()),
    )

    if row['Finalize'] == 'Yes':
      for path in (chk_f / CHECK_NAME).glob('*.wav'):
        shutil.copy(path, MOVE_TO_FOLDER_LOC / (folder_name.name+'_'+path.name))

