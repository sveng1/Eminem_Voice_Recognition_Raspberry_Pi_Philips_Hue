import sounddevice as sd
from scipy.io.wavfile import write
import datetime


sr = 48000
seconds = 9
frames = int(sr*seconds)

save_path = '/media/pi/TOSHIBA EXT/pi/audio_data/test'
i = 0

print('Sampling rate: {}, recording length: {} seconds'.format(sr, seconds))
print('Save under {}'.format(save_path))


print('Started recording')
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