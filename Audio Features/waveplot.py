import matplotlib.pyplot as plt
import librosa.display
plt.rcParams.update({'font.size': 16})

y, sr = librosa.load(librosa.util.example_audio_file())
plt.figure(figsize=(18, 7))
librosa.display.waveplot(y, sr=sr, x_axis='s')
print(sr)
plt.ylabel('Sampling Rate',fontsize=32)
plt.xlabel('Time (s)',fontsize=32)
plt.show()