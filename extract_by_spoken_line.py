
# import whisperx
# import gc 
# import pandas as pd


# # Uses a line to quickly pull a voice out.
# import subprocess, argparse
# from functools import partial
# from pathlib import Path
# import pandas as pd
# import spacy


# # parser = argparse.ArgumentParser()
# # parser.add_argument('input_path_lines' , help='Input path')
# # parser.add_argument('input_dir_speaker_outputs' , help='Input path')
# # # parser.add_argument('output_path', help='Output path')
# # parser.add_argument('line_to_check_for', help='Line to check for')
# # args = parser.parse_args()

# # input_path_lines = Path(args.input_path_lines)

# def check_for_given_line(input_path_lines, line_to_check_for):
#   rtm_values = pd.read_csv(
#     input_path_lines, delimiter='\s', header=None,
#     usecols=[5,8,11], names=['Start','Clip_Length','Speaker'],  
#     engine='python'
#   )
  

# nlp = spacy.load("en_core_web_lg")  # make sure to use larger package!

# bp = Path('/home/john/projects/utility/nemo_in_action/fmt_and_dia_trial/results/[bonkai77].Neon.Genesis.Evangelion.Episode.05.[BD.1080p.Dual-Audio.x265.HEVC.10bit].wav')
# INP_L = bp / 'diarization/Diarization_Formatted.csv'
# INP_SO = bp / 'clipped_audio'
# SL = nlp("But I'm used to it now")


# df = pd.read_csv(INP_L)
# # df = df[df['Clip_Over_Min_Secs'] == True]
# df['Similarity_Score'] = \
#   df['Highest_Likelihood_Line'] \
#     .apply(lambda x: nlp(x).similarity(SL))

# df[df['Similarity_Score'] > .9]
# if len(df) == 0:
#   print(
#     f'Could not find line {SL}'
#   )
# else:
#   best_fit_row = 


# df.merge(
#   df.groupby('Highest_Likelihood_Line')['Clip_Length'].max(),
  
# )

# df['Highest_Likelihood_Line'] = df['Highest_Likelihood_Line'].where(
#   df.duplicated('Highest_Likelihood_Line', keep='first')==False,
#   None
# )




# df_g

# df_g['index'] = df_g['index'].apply(lambda x: x)
# df_g = df_g.drop(columns='Clip_Length')

# df_g

# df_g
# type(df_g)

# df.index


# df.merge(df_g, how='left', left_index=True, right_index=True)
# df
# df

# df_g

#   #  ['Clip_Length'].max()

# df

# df_g

# df_g.reset_index()


# .sort_values('Similarity_Score', ascending=False)

# # OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.
# # python -m spacy download en_core_web_lg

# doc1 = nlp("I like salty fries and hamburgers.")
# doc2 = nlp("Fast food tastes very good.")

# # Similarity of two documents
# print(doc1, "<->", doc2, doc1.similarity(doc2))
# # Similarity of tokens and spans
# french_fries = doc1[2:4]
# burgers = doc1[5]
# print(french_fries, "<->", burgers, french_fries.similarity(burgers))
