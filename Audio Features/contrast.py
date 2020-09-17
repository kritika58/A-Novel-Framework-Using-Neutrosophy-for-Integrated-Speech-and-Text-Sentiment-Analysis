import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np

y, sr = librosa.load(librosa.util.example_audio_file())
S = librosa.feature.spectral_contrast(y, sr=sr, n_fft=2048, hop_length=512)
#S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()