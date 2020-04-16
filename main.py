import time
import numpy as np
import sounddevice as sd
from user import bridge_url, hue_user
from utils import audio2spectrogram, eminem_light
from hue_functions import get_light_state, get_connected_lights
from tensorflow.keras.models import load_model


# Load trained crnn model
model_path = 'eminem_model.h5'
model = load_model(model_path)
print('Loaded model from "{}".'.format(model_path))

# Set length of recording
time_step = 6

# Sample rate for recording
sr = 16000

# Length of recording
seconds = 3
frames = int(seconds * sr)
print('Sampling rate: {}, recording length: {} seconds'.format(sr, seconds))

# Initialize state and current hue and saturation value for each connected light
state = 'not eminem'
lights = get_connected_lights(bridge_url, hue_user)
previous_color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]

# Record audio
starttime=time.time()
print('Started recording')
try:
    while True:
        print('Time:', time.ctime())
        recording = sd.rec(frames=frames, samplerate=sr, channels=1)
        sd.wait()
        recording = recording.reshape(-1)

        # Transform audio to spectrograms
        spec = audio2spectrogram(audio=recording, sr=sr, audio_length=seconds, slice_len=seconds)

        # Predict class
        prediction = np.argmax(model.predict(spec)[0])

        # Use prediction to set light
        if not (state == 'not eminem' and prediction == 0):
            state, previous_color = eminem_light(bridge_url, hue_user, prediction, state, previous_color)
            time.sleep(time_step - ((time.time() - starttime) % time_step))
        print('Detected:', state)
        time.sleep(time_step - ((time.time() - starttime) % time_step))
except KeyboardInterrupt:
    print('Stopped recording')
