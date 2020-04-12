import os
import numpy as np
from tqdm import tqdm
from utils import load_audio_to_spectrogram


eminem_folder = ''
not_eminem_folder = ''

# sample rate of recorded audio files
sr = 48000

# length of recorded audio files in seconds
audio_length = 30

# data sample length
slice_len = 3

# To use the loop, each file must have same length (here 30 sec)
print('Loading recorded eminem audio')
eminem_spectrograms = []
for filename in tqdm(os.listdir(eminem_folder)):
    if filename.endswith('.wav'):
        images = load_audio_to_spectrogram(path=eminem_folder+filename, sr=sr,
                                           audio_length=audio_length, slice_len=slice_len)
        eminem_spectrograms.extend(images)
np.save(eminem_folder + 'eminem_spectrograms.npy', eminem_spectrograms)


print('Loading recorded not-eminem audio')
not_eminem_spectrograms = []
for filename in tqdm(os.listdir(not_eminem_folder)):
    if filename.endswith('.wav'):
        images = load_audio_to_spectrogram(path=not_eminem_folder+filename, sr=sr,
                                           audio_length=audio_length, slice_len=slice_len)
        not_eminem_spectrograms.extend(images)
np.save(not_eminem_folder + 'not_eminem_spectrograms.npy', not_eminem_spectrograms)