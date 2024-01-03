from nemo_diarization import diarize, generate_whisperx_transcription, clip_from_rttm, NoStdStreams
from rename_by_given_line import rename_by_line
import json, shutil, subprocess, argparse, gzip, os, time
from datetime import timedelta
from pathlib import Path
import pandas as pd, csv
from tqdm import tqdm
import random
from tqdm import trange

def diarize_clip_transcribe(audio_path, whisperx_path, base_output_directory,
                            name, window_length_in_sec, shift_length_in_sec, 
                            multiscale_weights, **kwargs):

  audio_path = Path(audio_path)
  dia_file_dir = (base_output_directory / 'diarization')
  clip_file_dir = (base_output_directory / 'clips')
  dia_file_dir.mkdir(exist_ok=True)
  clip_file_dir.mkdir(exist_ok=True)
  (dia_file_dir / 'temp').mkdir(exist_ok=True)
  (clip_file_dir / name).mkdir(exist_ok=True)
  with NoStdStreams():
    diarize(
      audio_path,
      (dia_file_dir / 'temp'),
      vad_path=whisperx_path,
      window_length_in_sec=window_length_in_sec,
      shift_length_in_sec=shift_length_in_sec,
      multiscale_weights=multiscale_weights
    )
  (dia_file_dir / 'temp/Diarization_Formatted.csv') \
    .rename(dia_file_dir / f"{name}.csv")
  shutil.rmtree(dia_file_dir / 'temp')
  
  clip_from_rttm(dia_file_dir / f"{name}.csv", audio_path,
                 clip_file_dir / f"{name}")
  
def name_and_filter(name, line, clipped_audio, diarization_csv):
  
  with NoStdStreams():
    rename_by_line(line_to_check_for=line, rename_folder_to=name, clipped_audio=clipped_audio, diarization_csv=diarization_csv, y=True)
  for folder in clipped_audio.glob('*'):
    if folder.name != name:
      shutil.rmtree(folder)
  for file in (clipped_audio / name).glob('*'):
    file.rename(file.parents[1] / file.name)
  shutil.rmtree(clipped_audio / name)

AUDIO_FILE = 'fmt_and_dia_trial/audio/[bonkai77].Neon.Genesis.Evangelion.Episode.08.[BD.1080p.Dual-Audio.x265.HEVC.10bit].wav'
OUT_DIR = Path('grid_search')
data_file_dir = (OUT_DIR / 'data_files')
dia_file_dir = (OUT_DIR / 'diarization')
clip_file_dir = (OUT_DIR / 'clips')
whisperx_t_path = data_file_dir / 'whisperx_transcription.csv'

OUT_DIR.mkdir(exist_ok=True)
data_file_dir.mkdir(exist_ok=True)

if not (whisperx_t_path.exists() and whisperx_t_path.with_suffix('.json').exists()):
  generate_whisperx_transcription(AUDIO_FILE, whisperx_t_path)

df_p = pd.DataFrame({
  'name': [
  #  'base', 'enhanced', 'enhanced_plus_top_end', 'long_window'
  'test_8'
  ],
  'window_length_in_sec': [
    # [1.9,1.2,0.5], 
    # [4,1.9,1.2,0.5,.25],
    # [5,4,1.9,1.2,0.5],
    # [3,1.5,1],
    [2.5,2,1.1,0.7],
  ],
  'shift_length_in_sec':[
    # [0.95,0.6,0.25],
    # [1.5,0.95,0.6,0.25,.1],
    # [2,1.5,0.95,0.6,0.25],
    # [0.95,0.6,0.25],
    [1.1,.8,0.5,0.2],
  ],
  'multiscale_weights': [
    # [1,1,1],
    # [1.3,1.2,1,1,.75],
    # [1.3,1.1,1,1,1],
    # [1,1,1], 
    [.9,.9,1,1]
  ],
})
  

def combine_audio_files_in_folder(in_folder, out_path, verbose=False):
  out_path = Path(Path(out_path).absolute())
  convert_pth = lambda x: str(Path(x).absolute().as_posix())
  text_file_path = in_folder / 'Files_to_combine.txt'

  media_paths = [i for i in in_folder.glob('*.wav')]
  media_path_len =  len(media_paths)
  if media_path_len == 0:
    return None

  range_func = trange if verbose else range
  for idx in range_func(len(media_paths)):
    media_path = media_paths[idx]
    if media_path.stem == '_combined':
      continue
    cmd = (
      f"ffmpeg -i "
      f"{convert_pth(media_path.with_suffix('.wav'))} "
      f"{convert_pth(media_path.with_suffix('.mp3'))} "
      "-codec:a libmp3lame -qscale:a 2 -y"
    )
    subprocess.check_call(
      cmd,
      shell=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL
    )

  media_paths = [i for i in in_folder.glob('*.mp3')]
  media_paths.sort()
  with open(text_file_path, 'w') as text_file:
    for file in media_paths:
      if file.stem == '_combined':
        continue
      file = convert_pth(file).replace('"',"'")
      text_file.write(f"file {file}")
      text_file.write('\n')

  cmd = (
    f"ffmpeg -f concat -safe 0 -i "
    f"{convert_pth(text_file_path)} " 
    f"-c copy {convert_pth(out_path.with_suffix('.mp3'))} -y"
  )
  if verbose:
    st = time.time()
  subprocess.check_call(
    cmd,
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
  )
  if verbose:
    print(f'Time for main command = {time.time()-st}')
  for path in media_paths:
    if path.stem != '_combined':
      path.unlink()
  if not verbose:
    text_file_path.unlink()

def combine_audio_files_in_folder_2(in_folder, out_path):
  out_path = Path(Path(out_path).absolute())
  convert_pth = lambda x: str(Path(x).absolute().as_posix())
  text_file_path = in_folder / 'Files_to_combine.txt'

  media_paths = [i for i in in_folder.glob('*.wav')]
  media_path_len =  len(media_paths)
  if media_path_len == 0:
    return None

  with open(text_file_path, 'w') as text_file:
    for file in media_paths:
      file = convert_pth(file).replace('"',"'")
      text_file.write(f"file {file}")
      text_file.write('\n')

  cmd = (
    f"ffmpeg -f concat -safe 0 -i "
    f"{convert_pth(text_file_path)} " 
    f"-c copy {convert_pth(out_path.with_suffix('.wav'))} -y",
  )
  print(cmd)
  # subprocess.check_call(
  #   cmd,
  #   shell=True,
  #   stdout=subprocess.DEVNULL,
  #   stderr=subprocess.DEVNULL
  # )

  cmd = (
    f"ffmpeg -i "
    f"{convert_pth(out_path.with_suffix('.wav'))} "
    f"{convert_pth(out_path.with_suffix('.mp3'))} "
    "-codec:a libmp3lame -qscale:a 2 ",
  )
  print(cmd)
  # subprocess.check_call(
  #   cmd,
  #   shell=True,
  #   stdout=subprocess.DEVNULL,
  #   stderr=subprocess.DEVNULL
  # )

  # text_file_path.unlink()
  # out_path.with_suffix('.wav').unlink()

def apply_uniform_dist(in_list, mean=0, std_dev=.5, sort_list=True, copy=100, min=.01):
  out_list = []
  for _ in range(copy):
    temp_list = [round(max(i+random.gauss(mean, std_dev),min),2) for i in in_list]
    if sort_list:
      temp_list.sort()
      temp_list.reverse()
    if len(set(temp_list)) != len(temp_list):
      add_list = [.01 * idx for idx in range(len(temp_list))]
      add_list.reverse()
      temp_list = [sum(x) for x in zip(temp_list, add_list)]
    out_list.append(temp_list)
  return out_list

# df_p = pd.DataFrame({
#   'name': [f'r_combo_{i}' for i in range(11,111)],
#   'window_length_in_sec': apply_uniform_dist([1.9,1.2,0.5], std_dev=.3, min=.2),
#   'shift_length_in_sec': apply_uniform_dist([0.95,0.6,0.25], std_dev=.3, min=.1),
#   'multiscale_weights': apply_uniform_dist([1,1,1], sort_list=False, min=.25),
# })

# for file in [data_file_dir / 'diarization_results.csv']:
#   if file.exists():
#     file.unlink()
# for folder in [dia_file_dir, clip_file_dir]:
#   if folder.exists():
#     shutil.rmtree(folder)

progress = tqdm(desc='Running Diarization Parameters', total=len(df_p.index))
with open(data_file_dir/ 'diarization_results.csv', 'w') as fp:
  writer=csv.writer(fp)
  writer.writerow(df_p.columns.tolist() + ['total_secs', 'time_elapsed'])

  for idx in df_p.index:
    start_time = time.time()
    row = df_p.loc[idx]
    # print(row)
    name = row['name']
    try:
      diarize_clip_transcribe(
        AUDIO_FILE,
        whisperx_t_path.with_suffix('.json'),
        OUT_DIR,
        **row.to_dict()
      )

      name_and_filter("Asuka", "You're coming with me.", 
                    clipped_audio=clip_file_dir/name,
                    diarization_csv=(dia_file_dir/name).with_suffix('.csv'))
      
      df_d = pd.read_csv(dia_file_dir / f"{name}.csv")
      df_p.at[idx, 'total_secs'] = df_d[df_d['Speaker'] == "Asuka"]['Clip_Length'].sum()
      df_p.at[idx, 'time_elapsed'] = time.time() - start_time 
      row_formatted = [
        str(i).replace(",","") if type(i) == list else i
        for i in df_p.loc[idx].tolist()
      ]
      writer.writerow(row_formatted)
      combine_audio_files_in_folder(
        clip_file_dir/name, (clip_file_dir/name) / '_combined.mp3')
    except:
      print(f'err row: {row}')
    progress.update(1)
