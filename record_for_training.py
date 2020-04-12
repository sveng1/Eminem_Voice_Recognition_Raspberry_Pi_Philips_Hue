import sounddevice as sd
from scipy.io.wavfile import write
import datetime


# Sampling rate for recording
sr = 48000

# Length of recording in seconds
seconds = 30
frames = int(sr*seconds)

# Path to save recordings
save_path = ''

print('Sampling rate: {}, recording length: {} seconds'.format(sr, seconds))
print('Save under {}'.format(save_path))

print('Started recording')
i = 0
try:
    while True:
        print(datetime.datetime.now())
        recording = sd.rec(frames=frames, samplerate=sr, channels=1)
        sd.wait()
        recording = recording.reshape(-1)
        write(filename=save_path+str(i)+'.wav', rate=sr, data=recording)
        i += 1
except KeyboardInterrupt:
    print('Stopped by user')