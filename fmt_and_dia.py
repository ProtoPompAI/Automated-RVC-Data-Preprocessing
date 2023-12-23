# Code wrapper to do both python files at once

import subprocess, argparse
from functools import partial
from pathlib import Path

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path' , help='Input path')
  parser.add_argument('output_path', help='Output path')
  parser.add_argument('-b', '--break_up_by', default=0, type=int, help='Break the audio up by X minutes. Output_file is converted to an output directory at the same location.')
  parser.add_argument('-k', '--keep_audio_seperated', action='store_true', help='If this option is set, processes audio without combing the outputs or breaking them apart.')

  s_call = partial(
    subprocess.check_call,
    shell=True,
    # stdout=subprocess.DEVNULL,
    # stderr=subprocess.DEVNULL
  )

  #  "/mnt/c/Users/John/Peer Downloads/anime/[bonkai77] Neon Genesis Evangelion DC [BD-1080p] [DUAL-AUDIO] [MULTI-SUB] [x265] [HEVC] [AAC] [10bit] {FILTERED}"

  args = parser.parse_args()
  in_path  = Path(args.input_path)
  out_path = Path(args.output_path)
  out_path.mkdir(exist_ok=True)
  Path(out_path / 'audio').mkdir(exist_ok=True)
  Path(out_path / 'results').mkdir(exist_ok=True)

  format_audio_command = f'python format_audio_for_inference.py "{in_path}" "{(out_path / "audio")}"'
  if args.break_up_by != 0:
    format_audio_command += f' -b {args.break_up_by}'
  if args.keep_audio_seperated:
    format_audio_command += ' -k'
  # format_audio_command += ' -m'
  
  # print(format_audio_command)
  print('='*5, 'Formating Audio Command', '='*5)
  print(format_audio_command)
  print('='*len('Formating Audio Command') + '='*12)
  s_call(format_audio_command)

  print('='*5, 'Diarization + Extraction',  '='*5)
  diarization_command = f'python nemo_diarization.py "{(out_path / "audio")}" "{(out_path / "results")}" -c'
  print('='*len('Diarization + Extraction') + '='*12)
  print(diarization_command)
  s_call(diarization_command)
