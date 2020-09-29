import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16})
y, sr = librosa.load(librosa.util.example_audio_file())
# librosa.feature.chroma_stft(y=y, sr=sr)
# # array([[ 0.974,  0.881, ...,  0.925,  1.   ],
# # [ 1.   ,  0.841, ...,  0.882,  0.878],
# # ...,
# # [ 0.658,  0.985, ...,  0.878,  0.764],
# # [ 0.969,  0.92 , ...,  0.974,  0.915]])

# # Use an energy (magnitude) spectrum instead of power spectrogram

# S = np.abs(librosa.stft(y))
# chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# chroma
# array([[ 0.884,  0.91 , ...,  0.861,  0.858],
# [ 0.963,  0.785, ...,  0.968,  0.896],
# ...,
# [ 0.871,  1.   , ...,  0.928,  0.829],
# [ 1.   ,  0.982, ...,  0.93 ,  0.878]])

# Use a pre-computed power spectrogram with a larger frame

S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

# array([[ 0.685,  0.477, ...,  0.961,  0.986],
# [ 0.674,  0.452, ...,  0.952,  0.926],
# ...,
# [ 0.844,  0.575, ...,  0.934,  0.869],
# [ 0.793,  0.663, ...,  0.964,  0.972]])

plt.figure(figsize=(15, 6))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='s')
plt.colorbar()
plt.ylabel('Pitch Class',fontsize=28)
plt.xlabel('Time (s)',fontsize=28)
plt.tight_layout()
plt.show()
