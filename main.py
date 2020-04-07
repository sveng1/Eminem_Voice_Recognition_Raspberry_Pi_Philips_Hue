import time
from tensorflow.keras.models import load_model
import sounddevice as sd
from utils import audio2spectrogram
from hue_functions import set_color_all


# Load trained crnn model
model_path = '/data/best_model_3_sec.h5'
model = load_model(model_path)


# Philips Hue bridge url and user id
bridge_url = 'http://192.168.0.42'
user = 'UEDVry7d0T4WYiNW58fA61pZ-dixxHkMhjtz3bQl'

# Values for time loop
time_step = 6
starttime=time.time()

# Values for recording
sr = 16000
seconds = 3
frames = int(seconds * sr)

try:
    while True:
        print(time.ctime())
        # Record audio
        recording = sd.rec(frames=frames, samplerate=sr, channels=1)
        sd.wait()
        recording = recording.reshape(-1)

        # Tranform audio to spectrograms
        spec = audio2spectrogram(recording)

        # Predict eminem or not eminem
        prediction = np.argmax(model.predict(spec)[0])

        # Use prediction to set light
        eminem_light(bridge_url, user, previous_color, prediction)

        time.sleep(time_step - ((time.time() - starttime) % time_step))
except KeyboardInterrupt:
    print('Stopped by user')