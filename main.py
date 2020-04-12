import time
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
from utils import audio2spectrogram, eminem_light
from user import bridge_url, hue_user
from hue_functions import get_light_state, get_connected_lights


model_path = 'model_recorded.h5'
model = load_model(model_path)
print('Loaded model from "{}".'.format(model_path))

time_step = 6
starttime=time.time()

sr = 48000
seconds = 3
frames = int(seconds * sr)
print('Sampling rate: {}, recording length: {} seconds'.format(sr, seconds))

# Get current hue and saturation value for each connected light
lights = get_connected_lights(bridge_url, hue_user)
previous_color = [[get_light_state(l)[key] for key in ['hue', 'sat']] for l in lights]
state = 'not eminem'

print('Started recording')
try:
    while True:
        print('Time:', time.ctime())
        # Record audio
        recording = sd.rec(frames=frames, samplerate=sr, channels=1)
        sd.wait()
        recording = recording.reshape(-1)

        # Tranform audio to spectrograms
        spec = audio2spectrogram(audio=recording, sr=48000, audio_length=seconds, slice_len=seconds)

        # Predict eminem or not eminem
        #prediction = np.random.randint(2, size=1)
        prediction = np.argmax(model.predict(spec)[0])

        # Use prediction to set light
        if not (state == 'not eminem' and prediction == 0):
            state, previous_color = eminem_light(bridge_url, hue_user, prediction, state, previous_color)
            time.sleep(time_step - ((time.time() - starttime) % time_step))
        print('Detected:', state)
        time.sleep(time_step - ((time.time() - starttime) % time_step))
except KeyboardInterrupt:
    print('Stopped by user')
