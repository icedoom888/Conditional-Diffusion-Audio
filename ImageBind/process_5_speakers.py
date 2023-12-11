from pydub import AudioSegment
import os
from moviepy.editor import concatenate_audioclips, AudioFileClip

def concatenate_audio_moviepy(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audio_clip_paths]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)

names = ['Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard', 'Magaret_Tarcher', 'Nelson_Mandela']

for name in names:
    data_path = f'/home/alberto/ImageBind/5_speakers/{name}'
    out_path = f'/home/alberto/ImageBind/5_speakers_p/{name}'

    os.makedirs(out_path, exist_ok=True)
    files = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    combined_sounds = []
    for i in range(len(os.listdir(data_path))-1):
        file_path = os.path.join(data_path, f'{i}.wav')
        combined_sounds.append(file_path)

        if i % 5 == 0:
            concatenate_audio_moviepy(combined_sounds, os.path.join(out_path, f'{i//5}.wav'))
            combined_sounds = []

    

