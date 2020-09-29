import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16})
y, sr = librosa.load(librosa.util.example_audio_file())

#display Spectrogram
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(18, 7))
librosa.display.specshow(Xdb, sr=sr, x_axis='s', y_axis='hz') 
#If to pring log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.ylabel('Frequency (Hz)',fontsize=28)
plt.xlabel('Time (s)',fontsize=28)
plt.colorbar()
plt.show()