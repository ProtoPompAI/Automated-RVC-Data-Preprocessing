# Formats audio from a file so it is ready for NeMo processing

import subprocess, shutil, mimetypes, argparse, re, traceback, zipfile, sys
from datetime import timedelta, datetime
from pathlib import Path
from functools import partial
from tqdm import tqdm
import wget
from nemo.utils.data_utils import resolve_cache_dir 

def get_s_call(verbose):
    if not verbose:
        s_call = partial(
            subprocess.check_call,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        s_call = partial(subprocess.call, shell=True)
    return s_call

convert_pth = lambda x: f'"{str(Path(x).absolute().as_posix())}"'
   
def convert_to_audio_input(in_path, out_path, time_stamp_in=None, time_stamp_out=None, break_up_by=0, keep_audio_seperated=False, keep_music=False, verbose=False):
    """
    Preps audio data for inference with FFMPEG.
    4 General Use Cases
    1) in_path is a file/directory and parameters set to defaults: Will convert file(s) to a diarization ready audio file
    2) break_up_by set: Will convert file(s) to a diarization ready audio directory. The out files will be clipped by a break_up_by mintues.
    3) keep_audio_seperated set true: Will convert files to a diarization ready audio directory. No timestamp editing / file combining will be perfomed.
    4) time_stamp_in & time_stamp_out set: Will convert file(s) to a diarization ready audio file clipped to a given timestamp


    ::in_path:: Input path. If it input path is a file, uses a single file.
        If the input path is a directory, will combine the files into a single file.
    ::out_path:: Output path. Must be a .wav file
    ::time_stamp_in:: ::time_stamp_out:: Can be set to none to convert all of the audio
        If these parameters are set, both values must be provided in format of HH:MM:SS 
    ::verbose:: If set to true, will print the FFMPEG output to console. Can be used to error check.
    ::break_up_by:: If set to a non-zero integer, will break up the audio by break_up_by mintues
    ::keep_audio_seperated:: If set to true, will not combine the audio outputs and will keep audio outputs original name in out_path directory.
    ::music_removal:: Bool to optionally use Vocal Remover to remove the music from an audio clip
    """
    
    # Combining audio inputs if the given path was a directory
    def get_media_paths(in_path):
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

    s_call = get_s_call(verbose)

    in_path = Path(in_path)
    out_path = Path(out_path)
    
    # Helpful logic variables
    inpath_is_dir = (in_path.suffix == '')
    outpath_is_dir = (out_path.suffix == '')
    outpath_is_wav = out_path.suffix != '.wav'
    time_stamp_in_set = not time_stamp_in is None
    time_stamp_out_set  = not time_stamp_out is None
    break_up_by_set = not (break_up_by is None or break_up_by == 0)

    # Error checking inputs ===
    if not out_path.parent.exists():
        raise FileNotFoundError(f'Could not the parent of the output file/directory: {out_path.parent.absolute()}')

    type_error_display = lambda x,y: f'Both {x} and {y} have been activated. Please choose one or neither to use.'
   
    if break_up_by_set: # Case 2: break_up_by_set is activated
        if keep_audio_seperated:
            raise ValueError(type_error_display('break_up_by', 'keep_audio_seperated'))
        elif time_stamp_in_set or time_stamp_out_set:
            raise ValueError(type_error_display('break_up_by', 'time_stamp_in/time_stamp_out.'))
        if not outpath_is_dir:
            raise ValueError('If break_up_by is set, the out_path must be a directory.')
    elif keep_audio_seperated: # Case 3: keep_audio_seperated is activated
        if time_stamp_in_set or time_stamp_out_set:
            raise ValueError(type_error_display('keep_audio_seperated', 'time_stamp_in/time_stamp_out'))
        if not inpath_is_dir:
            raise ValueError('If keep_audio_seperated is set, the in_path must be a directory.')
        if not outpath_is_dir:
            raise ValueError('If keep_audio_seperated is set, the out_path must be a directory.')
    elif time_stamp_in_set or time_stamp_out_set: # Case 4: time_stamp_in + time_stamp_out are activated
        if (not time_stamp_in_set) or (not time_stamp_out_set):
            raise ValueError('Time_stamp_in and time_stamp_out must either be provided together or not all at.')
        if not outpath_is_wav:
            raise ValueError('If time_stamp_in/time_stamp_out is set, the out_path must be a .wav file.')
    else: # Case 1: kwargs are kept at defaults
        if not outpath_is_wav:
            raise ValueError('If either break_up_by_set or keep_audio_seperated are not set, the out_path must be a .wav file.')
        
    if inpath_is_dir:
        get_media_paths(in_path) # Quick check to see if the input directory contains media files   

    # Helper functions ===
    def apply_vocal_removal(input_file, output_file, vocal_remover_version='6.0.0b2', verbose=False):
        """
        Uses github repository https://github.com/tsurumeso/vocal-remover/ to remove music from vocals.
        Will download the github repository to the NeMo cache directory if not found there.
        
        ::input_file:: input audio file
        ::output_file:: output audio file
        ::vocal_remover_version:: verison of the vocal remover
            if a value that is not found on https://github.com/tsurumeso/vocal-remover/releases/ is used,
            this function will result in error.
        """

        def generate_vr_path():
            path = Path(resolve_cache_dir()) / f'vocal-remover-v{vocal_remover_version}'
            path = path.with_name(path.name.replace('.','-'))
            return path
        
        def dl_vocal_removal():
            VOCAL_REMOVER_URL = (
                f'https://github.com/tsurumeso/vocal-remover/releases/download/'
                f'v{vocal_remover_version}/vocal-remover-v{vocal_remover_version}.zip'
            )
            out_folder = generate_vr_path()
            zip_file_name = out_folder.with_suffix('.zip')
            
            if not out_folder.exists():
                if not zip_file_name.exists():
                    f = wget.download(VOCAL_REMOVER_URL, str(zip_file_name.absolute().as_posix()), bar=None)
                else:
                    f = str(zip_file_name)
                with zipfile.ZipFile(f, 'r') as zip_ref:
                    zip_ref.extractall(out_folder)
                Path(f).unlink(True)
        
        # Applies vocal removal to remove music from audio file
        vocal_removal_folder = generate_vr_path()
        if not vocal_removal_folder.exists():
            dl_vocal_removal()

        s_call(
            f"cd {convert_pth(vocal_removal_folder / 'vocal-remover')} && "
            f"\"{sys.executable}\" {convert_pth(vocal_removal_folder / 'vocal-remover/inference.py')} " 
            f"--input {convert_pth(input_file)} --output_dir {convert_pth(output_file.parent)} "
            # "--postprocess " # Not functional in v6-0-0b2 " +
            "--tta "
            "--gpu 0 "
            "--batchsize 8"
            "--" 
        )
        get_end_file = lambda path, path_end: path.with_name(path.stem + f'{path_end}')
        get_end_file(input_file, '_Vocals.wav').rename(output_file)
        get_end_file(input_file, '_Instruments.wav').unlink()        

    # Converting from a format such as a .mp4 to .wav
    def convert_to_audio(in_path, out_path):
        s_call(
            f"ffmpeg -i {convert_pth(in_path)} -ab 160k -ac 2 -ar 44100 -vn "
            f'{convert_pth(out_path)} -y'
        )
   
   # Nemo only works with mono audio, so this coverts stero to mono
    def convert_to_mono(in_path, out_path):
        s_call(
            f"ffmpeg -i {convert_pth(in_path)} -ac 1 "
            f'{convert_pth(out_path)} -y'
        )

    # Function that is called to cut the audio by timestamps if required
    def cut_to_timestamp(start_time, end_time, in_path, out_path):
        s_call(
            f"ffmpeg -i {convert_pth(in_path)} -ss {start_time} -to {end_time} "
            f"-c:v copy -c:a copy {convert_pth(out_path)} -y", shell=True
        )

    # Fully converts a sigle file to audio input 
    def convert_file_to_audio_input(in_path, out_path, temp_out_folder):
        # Creating tempory files, FFMPEG may not always work with in place operations
        temp_audio_paths = {
            'to_wav': temp_out_folder / (out_path.stem + '_conv_to_wav.wav'),
            'to_mono': temp_out_folder / (out_path.stem + '_conv_to_mono.wav'),
            'to_removed_music': temp_out_folder / (out_path.stem + '_removed_music.wav')
        }
        out_path = out_path.with_suffix('.wav') # Ensures that a .wav file is the output file

        convert_to_audio(in_path, temp_audio_paths['to_wav'])
        convert_to_mono(temp_audio_paths['to_wav'], temp_audio_paths['to_mono'])
        if keep_music==False:
            apply_vocal_removal(temp_audio_paths['to_mono'], temp_audio_paths['to_removed_music'], verbose=verbose)     
            temp_audio_paths['to_removed_music'].rename(out_path)
        else:
            temp_audio_paths['to_mono'].rename(out_path)

        for file_path_key in list(temp_audio_paths):
            temp_audio_paths[file_path_key].unlink(True)

    def convert_and_combine_wav(in_path, out_path, temp_out_folder):
        media_paths = get_media_paths(in_path)
 
        wav_files_to_combine = []
        for idx in tqdm(range(len(media_paths)), desc="Converting files to proper audio"):
            temp_out_path = temp_out_folder / f'temp_audio_file_{idx}.wav'
            convert_file_to_audio_input(media_paths[idx], temp_out_path, temp_out_folder)
            wav_files_to_combine.append(str(convert_pth(temp_out_path)))
    
        # Making a file with all of the audio files for ffmpeg 
        with open(temp_out_folder / 'Files_to_combine.txt', 'w') as text_file:
            for wav_file in wav_files_to_combine:
                wav_file = wav_file.replace('"',"'")
                text_file.write(f"file {wav_file}")
                text_file.write('\n')
    
        # Using FFMPEG to combine all of the wav files
        s_call(
            f"ffmpeg -f concat -safe 0 -i "
            f"{convert_pth(temp_out_folder / 'Files_to_combine.txt')} " 
            f"-c copy {convert_pth(out_path)} -y"
        )

    # Convert python timestamp to seconds
    def conv_to_sec(x):
        return (x.hour * 60 * 60) + (x.minute * 60) + x.second + float(f".{x.microsecond}")

    # Formatting seconds for command line execution as well as file names
    def fmt_sec(seconds, file_name_style=False):
        hours        = int(seconds // (60*60))
        minutes      = int(seconds % (60*60) // 60)
        seconds_disp = int(seconds % (60*60) % 60)
        miliseconds = seconds % 1
        if file_name_style == False:
            if miliseconds != 0:
                return f"{hours:02d}:{minutes:02d}:{seconds_disp:02d}.{str(miliseconds).split('.')[-1]}"
            else:
                return f"{hours:02d}:{minutes:02d}:{seconds_disp:02d}"
        else:
            return f"{hours:02d}-{minutes:02d}"

    # Ensures that all temporary files are erased on exit
    def clean_temp_outputs(temp_out_dir):
        shutil.rmtree(temp_out_dir)

    def get_clip_len(file_path):
        clip_len = str(subprocess.check_output(
            f'ffmpeg -i {convert_pth(file_path)} 2>&1 | grep "Duration"', shell=True
        ))
        clip_len = re.search(r'Duration: ([0-9:\.]*),', clip_len).group(1)
        clip_len = datetime.strptime(clip_len, '%H:%M:%S.%f')
        clip_secs = conv_to_sec(clip_len)
        return clip_len, clip_secs

    # Executing audio processing ===

    if outpath_is_dir:
        out_path.mkdir(exist_ok=True)
        temp_out_dir = out_path / 'temp_audio_dir'
        temp_out_dir.mkdir(exist_ok=True)
    else:
        temp_out_dir = out_path.parent / 'temp_audio_dir'
        temp_out_dir.mkdir(exist_ok=True)

    try:
        if keep_audio_seperated: # Simple case of keeping the audio seperated
            # tqdm(range(len(media_paths)), desc="Converting files to proper audio")
            media_paths = get_media_paths(in_path)
            for idx in tqdm(range(len(media_paths)), desc="Processing input files individually"):
                convert_file_to_audio_input(media_paths[idx], (out_path / media_paths[idx].name), temp_out_dir)
        else:
            # Generating a wav temporary file, then deciding what to do with it based on given parameters
            temp_out_file = temp_out_dir / 'All_Audio_Combined_Temp.wav'
            if inpath_is_dir:
                convert_and_combine_wav(in_path, temp_out_file, temp_out_dir)
            else:
                convert_file_to_audio_input(in_path, temp_out_file, temp_out_dir)
            if time_stamp_in_set and time_stamp_out_set:
                cut_to_timestamp(time_stamp_in, time_stamp_out, temp_out_file, out_path)
            elif break_up_by_set:
                clip_len, clip_secs = get_clip_len(temp_out_file)
                if clip_secs < break_up_by * 60:
                    raise ValueError(
                        f'Break up by {break_up_by} minutes is less than the '
                        f'duration of the clip: ({clip_len.strftime("%H:%M:%S.%f")}).\n'
                    )
                else:
                    desc = f"Extracting {break_up_by} minute segments"
                    for sec_start in tqdm(range(0, int(clip_secs) + 1, break_up_by * 60), desc=desc):
                        sec_end = min(sec_start + break_up_by * 60, clip_secs)
                        cut_to_timestamp(
                            fmt_sec(sec_start), fmt_sec(sec_end), temp_out_file,
                            out_path / (f"{fmt_sec(sec_start, True)}_to_" +
                                        f"{fmt_sec(sec_end, True)}.wav")
                        )
            else:
                temp_out_file.rename(out_path)
        clean_temp_outputs(temp_out_dir)
    except Exception: # Cleans up outputs in case of error
        clean_temp_outputs(temp_out_dir)
        traceback.print_exc()
        if out_path.exists():
            if outpath_is_dir:
                shutil.rmtree(out_path)
            else:
                out_path.unlink(True)
        raise ValueError()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path' , help='Input path')
    parser.add_argument('output_path', help='Output path')
    parser.add_argument('-s', '--start_time', required=False, help='Clip the file from start time.')
    parser.add_argument('-e', '--end_time'  , required=False, help='Clip the file ending at end time.')
    parser.add_argument('-v', '--verbose'   , action='store_true', help='Display full ffmpeg output on the command line.')
    parser.add_argument('-b', '--break_up_by', default=0, type=int, help='Break the audio up by X minutes. Output_file is converted to an output directory at the same location.')
    parser.add_argument('-k', '--keep_audio_seperated', action='store_true', help='If this option is set, processes audio without combing the outputs or breaking them apart.')
    parser.add_argument('-m', '--keep_music', action='store_true', help='Option to skip music removal') 
    args = parser.parse_args()
    convert_to_audio_input(
        in_path=args.input_path,
        out_path=args.output_path,
        time_stamp_in=args.start_time,
        time_stamp_out=args.end_time,
        break_up_by=args.break_up_by,
        keep_audio_seperated=args.keep_audio_seperated,
        keep_music=args.keep_music,
        verbose=args.verbose
    )
