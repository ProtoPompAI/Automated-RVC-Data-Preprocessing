import pandas as pd
import subprocess
import time
from itertools import product

default_params = {
  "sr": 44100,
  "n_fft": 2048,
  "hop_length": 1024,
  "batchsize": 4,
  "cropsize": 256,
}

param_list = {
  k: [int(v*.5),int(v*1.5)]
  for k, v in default_params.items()
}
param_list['batchsize'] = [2,4,8,16]

base_call = [
  "cd", '"/home/john/.cache/torch/NeMo/NeMo_1.21.0rc0/vocal-remover-v6-0-0b2/vocal-remover/"', "&&",
  '"/home/john/anaconda3/envs/nemo/bin/python"',
  '"/home/john/.cache/torch/NeMo/NeMo_1.21.0rc0/vocal-remover-v6-0-0b2/vocal-remover/inference.py"',
  "--input", 
  '"/home/john/projects/utility/nemo_in_action/tools/short_clip.wav"',
  "--output_dir", '"/home/john/projects/utility/nemo_in_action/temp_audio_dir"',
  "--tta", "--gpu", "0",
]
df = pd.DataFrame()

for vals in product(*param_list.values()):
  st = time.time()
  val_dict = dict(zip(param_list, vals))
  bg2 = base_call
  for k, v in default_params.items():
    bg2.append(f"--{k}")
    bg2.append(str(v))
  subprocess.call(" ".join(bg2), shell=True)

  val_dict['time'] = time.time() - st
  df = pd.concat([df, pd.DataFrame(val_dict,index=[0])], ignore_index=True)

df.to_csv('TIME_TEST_2.csv',index=False)

# def time_it(**kwargs):
#   st = time.time()
#   base_call = [
#     "cd", '"/home/john/.cache/torch/NeMo/NeMo_1.21.0rc0/vocal-remover-v6-0-0b2/vocal-remover/"', "&&",
#     '"/home/john/anaconda3/envs/nemo/bin/python"',
#     '"/home/john/.cache/torch/NeMo/NeMo_1.21.0rc0/vocal-remover-v6-0-0b2/vocal-remover/inference.py"',
#     "--input", '"/home/john/projects/utility/nemo_in_action/Trial_Test_213/temp_audio_dir/[bonkai77].Neon.Genesis.Evangelion.Episode.09.[BD.1080p.Dual-Audio.x265.HEVC.10bit]_conv_to_wav.wav"',
#     "--output_dir", '"/home/john/projects/utility/nemo_in_action/temp_audio_dir"',
#     "--tta", "--gpu", "0",
#   ]
#   for k, v in kwargs.items():
#     bg2.append(f"--{k}")
#     bg2.append(str(v))
#   subprocess.call(" ".join(bg2), shell=True)
#   time_taken = time.time() - st
#   print(kwargs)
#   print(time_taken)

# test_params = {
#   "sr": 55125,
#   "n_fft": 2560,
#   "hop_length": 768,
#   "batchsize": 4,
#   "cropsize": 192,
# }

# time_it(**test_params)

df = pd.read_csv('TIME_TEST_2.csv')
df.sort_values('time')

df