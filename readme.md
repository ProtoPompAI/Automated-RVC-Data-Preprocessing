[ProtoPomp Website](https://protopomp.com)

[ProtoPomp YouTube](https://youtube.com/@protopomp)

# Automated RVC Data Processing
*Data preprocessing for RVC can be as easy as a single command.*
This is an opensource Python tool to automate RVC data preprocessing.

## Primary Toolkits used

## Basic Instructions
* Git clone this repository `git clone https://github.com/ProtoPompAI/Automated-RVC-Data-Preprocessing.git`
* Create a Python 3.10 virtual environment i.e. `conda create -n Automated-RVC-Data python=3.10 && conda activate Automated-RVC-Data`
  * This repository was developed in WSL. If unresolvable issues occur in Windows, it is recommended to use WSL or Linux instead.
* Install requirements within the new environment `pip install -r requirements.txt`
* Run the program by opening a command window and running the program `python preprocess_data.py INPUT_AUDIO_DIRECTORY OUTPUT_LOCATION`
  * The `INPUT_AUDIO_DIRECTORY` will be converted into workable audio with FFmpeg. Video files as well as audio files can be used as input into the program.
* Use `python preprocess_data.py --help` to see the different ways to process raw input data.
* It is highly recommended to use audio files that are at most 25 minutes long. Splitting long audio can be done with the `-b` command line argument.

## Optional Functionality
* By modifying the Excel file `Specified_Lines.xlsx` and adding the command line arguments `--specification_file` and `--speaker_label`, the command line program can extract a specific speaker out into a new folder. This gives the data the ability to be directly fed into training an RVC or removing background audio.

## Implementation Notes
*Want to understand or modify this repository? Helpful details are below.*
* What tools are under the hood? What's the use of each tool?
  * [NeMo](https://github.com/NVIDIA/NeMo). Diarization.
  * [WhisperX](https://github.com/m-bain/whisperX). Transcription and VAD Generation.
  * [FFmpeg](https://github.com/FFmpeg/FFmpeg). Manipulating audio files.
  * [Spacy](https://github.com/explosion/spaCy). Allows for truthy matches when comparing audio lines. Used in the optional speaker extraction to compare lines written in the Excel file and the WhisperX transcription.
* Why both NeMo and WhisperX? Both NeMo and WhisperX feature transcription as well as diarization capabilities.
  * After rigorous testing, it became clear in my experiments that NeMo had consistently better results in diarization tasks. In the same vein, WhisperX had consistently better results in pure transcription tasks. In fact, the VAD is done by WhisperX rather than the default NeMo tools.
* Are the settings for NeMo and WhisperX changed from default? If so, why?
  * WhisperX settings are kept to defaults.
  * For NeMo, the clustering diarizer and the domain_type of meeting are used. These default settings can be found [here](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml). All defaults are kept the same, apart from settings to allow NeMo to process longer audio files. These changes in settings have a large impact on performance, making NeMo diarization use more resources and take longer. However, without these setting changes, NeMo will often only pick up a single speaker in longer audio files.
    * ![Nemo settings modifications](img/NeMo_updated_settings.png)