# Code wrapper to do both python files at once

import subprocess, argparse, sys
from pathlib import Path

def opt_add(cmd_alias, cmd_arg):
  if cmd_arg not in [None, False]:
    if type(cmd_arg) == bool:
      return f"{cmd_alias}"
    else:
      return f"{cmd_alias} {cmd_arg}"

def format_path(path_name):
  return str(path_name.absolute().as_posix())
    
def format_script_name(file_name):
  return format_path(Path(__file__).parent / file_name)

def call_child_script(command):
  def filter_out_none(input_list):
    return [x for x in input_list if x is not None]
  subprocess.check_call(filter_out_none(command))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # Command line arguments for all files
  parser.add_argument('input_path' , help='Input path')
  parser.add_argument('output_path', help='Output path')
  parser.add_argument('-rf', '--redo_audio_formatting', action='store_true', help='If this option is set, redo the audio formatting if already completed')
  parser.add_argument('-rd', '--redo_diarization', action='store_true', help='If this option is set, redo the diarization if already completed')


  # Command line arguments for rename_speaker_by_line.py
  parser.add_argument('-sf', '--specification_file', required=False,
                      help='Excel file that specifies the speakers through a line.')
  parser.add_argument('-sl', '--speaker_label',required=False,
                      help='Speaker label. Needs to be present in the Excel specification file.')
  
  # Command line arguments for format_audio_for_inference.py
  parser.add_argument('-b', '--break_up_by', default=0, type=int,
                      help='Break the audio up by X minutes. Output_file is converted to an output directory at the same location.')
  parser.add_argument('-k', '--keep_audio_separated', action='store_true',
                      help='If this option is set, processes audio without combing the outputs or breaking them apart.')
  parser.add_argument('-st', '--start_time', required=False,help='Clip the file from start time.')
  parser.add_argument('-et', '--end_time'  , required=False, help='Clip the file ending at end time.')
  parser.add_argument('-v', '--verbose', action='store_true',
                      help='Display full ffmpeg output on the command line.')

  # =====

  args = parser.parse_args()
  Path(args.output_path).mkdir(exist_ok=True)
  format_audio_output_path = Path(args.output_path) / "formatted_audio"
  diarization_output_path = Path(args.output_path) / "diarization"
  if args.speaker_label is not None:
    speaker_clip_output_path = Path(args.output_path) / args.speaker_label
  else:
    speaker_clip_output_path = None


  format_audio_command = [
    sys.executable,
    format_script_name("format_audio_for_inference.py"),
    args.input_path,
    format_audio_output_path,
    opt_add('-b', args.break_up_by),
    opt_add('-k', args.keep_audio_separated),
    opt_add('-st', args.start_time),
    opt_add('-et', args.end_time),
    opt_add('-v', args.verbose),
  ]

  diarization_command = [
    sys.executable,
    format_script_name("nemo_diarization.py"),
    format_audio_output_path,
    diarization_output_path,
    '-c'
  ]

  rename_speaker_command = [
    sys.executable,
    format_script_name('rename_speaker_by_line.py'),
    diarization_output_path,
    args.specification_file,
    args.speaker_label,
    speaker_clip_output_path,
  ]

  if args.redo_audio_formatting or (not format_audio_output_path.exists()):
    print('='*5, 'Formatting raw audio', '='*5)
    call_child_script(format_audio_command)
    print('='*5, len('Formatting raw audio') * '=', '='*5, sep='=')
  else:
    print('Skipping formatting raw audio, use -rf to rerun audio formatting.')

  if args.redo_diarization or (not diarization_output_path.exists()):
    print('='*5, 'Diarization', '='*5)
    call_child_script(diarization_command)
    print('='*5, len('Diarization') * '=', '='*5, sep='=')
  else:
    print('Skipping diarization, use -rd to rerun diarization.')

  if (args.specification_file is not None) and (args.speaker_label is not None):
    print('='*5, f'Extracting speaker: {args.speaker_label}', '='*5)
    call_child_script(rename_speaker_command)
    print('='*5, len(f'Extracting speaker: {args.speaker_label}'), '='*5, sep='=')
  else:
     print('Skipping speaker clip extraction, specify -sf and -sl to extract speaker.')
