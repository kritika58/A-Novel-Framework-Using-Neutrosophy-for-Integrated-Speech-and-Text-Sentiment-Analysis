import matplotlib.pyplot as plt
import librosa.display

y, sr = librosa.load(librosa.util.example_audio_file())
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr, x_axis='s')

plt.show()