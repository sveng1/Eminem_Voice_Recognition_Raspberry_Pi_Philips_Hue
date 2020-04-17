# Eminem Voice Recognition with Raspberry Pi and Philips Hue

A while ago I had an incident. A friend and I were relaxing at my place, music was flowing from Spotify. A song comes on and I ask "Is that Eminem?". My friends says "yeah I think it's Eminem", and I'm like "No, I'm not sure it's him", and we discuss back and forth, but eventually we have to go check on her phone to find out whether or not it is Eminem singing. 

It was a very uncomfortable situation, and to make sure it doesn't happen again, I have now set up a system in my apartment that tells me whenever it is Eminem singing or rapping on my speaker. 

A voice recognition model trained on Eminem's voice and other sounds is running on a Raspberry Pi, and every time it detects Eminem's voice, it changes the Philips Hue light bulps in my apartment to light blue (because, as everyone knows, light blue is Eminem's favourite color). Now I can finally relax and don't have to embarrass myself by having to check on my phone if the music playing is really Eminem.

### Set up Raspberry Pi
The Raspberry Pi used is a 3b+ model with Raspbian Buster. <br>
The USB microphone is just a standard cheap omnidirectional microphone. The microphone should be visible in python with `sounddevice.query_devices()` . <br>
To be able to record with a sampling rate of 16khz, a file .asoundrc was created in the home directory containing the following

```
pcm.!default {
  type plug
  slave {
    pcm "hw:1,0"
  }
}
ctl.!default {
    type hw
    card 1
}
```

### Set up Philips Hue
A Philips Hue starter kit with three light bulbs and a bridge is used. <br>
A guide to use the API is here: https://developers.meethue.com/develop/get-started-2/. I also go through the steps here: https://github.com/sveng1/Philips_Hue/blob/master/philips_hue_example.ipynb. <br>
Basic functions for controlling the lights are in hue_functions.py

### Collecting data
Run record_for_training.py on the Raspberry Pi with a USB microphone. <br>
Two-three hours of Eminem singing and two-three hours of other sounds (silence, music, talking, etc.) were recorded as 30 seconds long .wav files and saved in two separate folders.

### Processing audio data
Run create_spectrograms.py. <br>
This script opens the audio files, splits them into parts of 3 seconds and transforms them to mel scaled short-time Fourier transform spectrograms and saves these. <br>
The hyperparameters used in processing are the same as in Nasrulla and Zhao (2019)([article](https://arxiv.org/pdf/1901.04555.pdf), [github](https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/src/utility.py))

### Training recognition model
Run train_model.py <br>
This imports a convolutional recurrent neural network defined in model.py. <br>
The model is trained on the spectrograms.
The hyperparameters used in creating and training the model are mostly the same as in Nasrulla and Zhao (2019) ([article](https://arxiv.org/pdf/1901.04555.pdf), [github](https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/src/models.py)).

### Detecting Eminem and controlling Philips Hue lights
Run main.py on the Raspberry Pi with a USB microphone. <br>
Every six seconds, three seconds of audio is recorded, transformed to a spectrogram and sent to the trained CRNN model for prediction. <br>
The prediction result is then used to control the Philips Hue lights using the function eminem_lights.
