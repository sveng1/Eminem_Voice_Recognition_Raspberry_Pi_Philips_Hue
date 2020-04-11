import librosa
import numpy as np
from tqdm import tqdm
import os
from hue_functions import set_color, set_color_all, get_connected_lights, get_light_state
from tensorflow.keras.utils import to_categorical
from scipy.signal import resample


def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


def audio2spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512, slice_len=3):
    """
    Splits audio into smaller chunks of length slice_len and creates a mel
    scale spectrogram for each of these. Returns a list of spectrograms reshaped
    into (n_mels, time (frames), 1)
    :param audio: array, audio
    :param sr: int, sampling rate
    :param n_mels: int, number of mel bins
    :param n_fft: int, fast fourier transform window size
    :param hop_length: int, hop length
    :param slice_len: int, length of audio slice in seconds
    :return: images: list, spectrogram arrays of each slice in audio
    """

    # Reshape audio array
    if len(audio.shape) > 1:
        audio = audio.reshape(-1)

    # Resample to sampling rate of 16000
    if sr != 16000:
        audio = resample(audio, slice_len*16000)
        sr = 16000

    # Check if audio length is divible by chosen length (seconds * sr)
    remainder = audio.shape[0] % (sr * slice_len)

    slices = chunks(audio, sr * slice_len)

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

    if remainder != 0:
        print('removing last spectrogram frame as it is too short')
        images = images[:-1]

    images = np.array(images)
    return images


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

    images = audio2spectrogram(x, sr=sr, n_mels=n_mels, n_fft=n_fft,
                               hop_length=hop_length, slice_len=slice_len)

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


def eminem_light(bridge_url, user, prediction, state, previous_color):
    """
    Changes light depending on prediction
    :param bridge_url: str, hue bridge url
    :param user: str, hue user id
    :param prediction: int, 0 or 1, prediction from model
    :param state: str, 'eminem' or 'not eminem'
    :param previous_color: list, color for each light ex: [[hue,sat],[hue,sat]]
    :return: state, previous_color
    """

    lights = get_connected_lights(bridge_url, user)

    # if it is eminem
    if prediction == 1:
        # if it was also eminem before
        if state == 'eminem':
            color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]
            # If it was already eminem and the color was changed in the mean time
            if color != [[38749, 162], [38749, 162], [38749, 162]]:
                previous_color = color
                set_color_all(bridge_url, user, hue=38749, sat=162)
        # if it was not eminem before
        elif state == 'not eminem':
            previous_color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]
            set_color_all(bridge_url, user, hue=38749, sat=162)
            state = 'eminem'

    elif prediction == 0:
        for i in range(len(lights)):
            set_color(light=lights[i], hue=previous_color[i][0], sat=previous_color[i][1])
            state = 'not eminem'

    return state, previous_color
