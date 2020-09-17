import glob  
import os  
import librosa  
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
print("\n -------------import completed-------------\n")

def extract_feature(file_name): 
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features= np.empty((0,193))  
    for sub_dir in enumerate(sub_dirs):
        print("sub_dir:", sub_dir[1])
        sub_dirs_2=os.listdir(str(main_dir+sub_dir[1]))
        for sd in enumerate(sub_dirs_2):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir[1], sd[1], file_ext)):
                print(fn)
                try:
                    mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                    print(fn)
                except Exception as e:
                    print ("Error encountered while parsing file: ", fn)
                    continue
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
    return np.array(features)

main_dir = "C:\\Users\\lenovo\\Desktop\\FINAL PROJECT\\LibriSpeech\\dev-clean\\"
sub_dirs=os.listdir(main_dir)  
print(sub_dirs)
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features = parse_audio_files(main_dir,sub_dirs)  
print("done features")  
np.save('dev_clean_features',features)  