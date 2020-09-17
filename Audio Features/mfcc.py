import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
y, sr = librosa.load(librosa.util.example_audio_file())

mfccs = librosa.feature.mfcc(y, sr=sr)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

#display waveplot
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)
plt.show()

#display Spectrogram
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
#If to pring log of frequencies  
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()