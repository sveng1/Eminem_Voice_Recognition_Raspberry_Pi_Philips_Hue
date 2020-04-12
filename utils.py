import librosa
import numpy as np
from scipy.signal import resample
from hue_functions import set_color, set_color_all, get_connected_lights, get_light_state


def chunks(l, n):
    """
    Splits list into chunks of size n
    :param l: list, array
    :param n: int, length of chunks
    :return: list of equally sized lists
    """
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


def audio2spectrogram(audio, sr, audio_length, slice_len, n_mels=128, n_fft=2048, hop_length=512):
    """
    Splits audio into smaller chunks of length slice_len and creates a mel
    scale spectrogram for each of these. Returns a list of spectrograms reshaped
    into (n_mels, time (frames), 1)
    :param audio: array, audio
    :param sr: int, sampling rate
    :param audio_length: int, length of recorded audio in seconds
    :param slice_len: int, length of audio slice in seconds
    :param n_mels: int, number of mel bins
    :param n_fft: int, fast fourier transform window size
    :param hop_length: int, hop length
    :return: images: list, spectrogram arrays of each slice in audio
    """

    # Reshape audio array
    if len(audio.shape) > 1:
        audio = audio.reshape(-1)

    # Resample to sampling rate of 16000
    if sr != 16000:
        audio = resample(audio, audio_length*16000)
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


def load_audio_to_spectrogram(path, sr, audio_length, slice_len, n_mels=128, n_fft=2048, hop_length=512):
    """
    Loads audio, splits it into smaller chunks of length slice_len and creates a mel
    scale spectrogram for each of these. Returns a list of spectrograms reshaped
    into (n_mels, time (frames), 1)
    :param path: str, path to audio file
    :param sr: int, sampling rate
    :param audio_length: int, input length in seconds
    :param slice_len: int, length of audio slice in seconds
    :param n_mels: int, number of mel bins
    :param n_fft: int, fast fourier transform window size
    :param hop_length: int, hop length
    :return: images: list, spectrogram arrays of each slice in audio
    """

    x, sr = librosa.load(path, sr=sr)

    images = audio2spectrogram(audio=x, sr=sr, audio_length=audio_length, n_mels=n_mels, n_fft=n_fft,
                               hop_length=hop_length, slice_len=slice_len)

    return images


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

    # predicted eminem
    if prediction == 1:
        # Check if eminem was previously predicted and the color was changed in the mean time
        if state == 'eminem':
            color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]
            if color != [[38749, 162], [38749, 162], [38749, 162]]:
                previous_color = color
                set_color_all(bridge_url, user, hue=38749, sat=162)
        # If eminem was not predicted previously, save current color and change color
        elif state == 'not eminem':
            previous_color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]
            set_color_all(bridge_url, user, hue=38749, sat=162)
            state = 'eminem'

    # predicted not eminem, change each light back to its previous color
    elif prediction == 0:
        for i in range(len(lights)):
            set_color(light=lights[i], hue=previous_color[i][0], sat=previous_color[i][1])
            state = 'not eminem'

    return state, previous_color
