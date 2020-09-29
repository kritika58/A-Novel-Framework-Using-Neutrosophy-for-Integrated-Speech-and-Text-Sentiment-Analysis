import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
y, sr = librosa.load(librosa.util.example_audio_file())

mfccs = librosa.feature.mfcc(y, sr=sr)
print(mfccs.shape)
# Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='s')
plt.ylabel('Sampling Rate',fontsize=16)
plt.xlabel('Time (s)',fontsize=16)
plt.show()