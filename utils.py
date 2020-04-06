import librosa
import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.utils import to_categorical



def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


def load_audio_to_spectrogram(path, sr=16000, n_mels=128, n_fft=2048, hop_length=512, slice_len=3):
    """
    Loads audio, splits it into smaller chunks of length slice_len and creates a mel
    scale spectrogram for each of these. Returns a list of spectrograms reshaped
    into (n_mels, time (frames), 1)
    :param path: str, path to audio file
    :param sr: int, sampling rate
    :param n_mels: int, number of mel bins
    :param n_fft: int, fast fourier transform window size
    :param hop_length: int, hop length
    :param slice_len: int, length of audio slice in seconds
    :return: images: list, spectrogram arrays of each slice in audio
    """

    x, sr = librosa.load(path, sr=sr)

    # Slice into parts of slice_len seconds
    slices = chunks(x, sr * slice_len)

    images = []

    for i in range(len(slices)):
        S = librosa.feature.melspectrogram(slices[i],
                                           sr=sr,
                                           n_mels=n_mels,
                                           n_fft=n_fft,
                                           hop_length=hop_length)
        log_S = librosa.amplitude_to_db(S, ref=1.0)
        log_S = log_S.reshape(log_S.shape[0], log_S.shape[1], 1)
        images.append(log_S)
    images = np.array(images[:-1])

    return images


def load_train_data(one_folder, zero_folder, sr=16000, n_mels=128, n_fft=2048, hop_length=512, slice_len=3):
    """
    Loads all training data from two folders using load_and_preprocess
    :param one_folder: path to first folder
    :param zero_folder: path to second folder
    :param sr: int, sampling rate
    :param n_mels: int, number of mel bins
    :param n_fft: int, fast fourier transform window size
    :param hop_length: int, hop length
    :param slice_len: int, length of audio slice in seconds
    :return: X, y: training data and labels
    """
    # Load all from class one
    print('Loading data from class one')
    one = []
    for filename in tqdm(os.listdir(one_folder)):
        if filename.endswith('.wav'):
            images = load_and_preprocess(one_folder + filename, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length, slice_len=slice_len)
            one.extend(images)

    # Load all from class zero
    print('Loading data from class zero')
    zero = []
    print('Loading data from class zero')
    for filename in tqdm(os.listdir(zero_folder)):
        if filename.endswith('.wav'):
            images = load_and_preprocess(zero_folder + filename, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length, slice_len=slice_len)
            zero.extend(images)

    # Concatenate images
    X = np.array(one + zero)

    # Create labels
    y = np.array([1] * len(one) + [0] * len(zero))
    y = to_categorical(y)

    return X, y
