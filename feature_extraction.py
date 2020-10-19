#Feature Extraction code for a particular folder

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
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    print("MFCC SHAPE\t", mfccs.shape)
    print("CHROMA SHAPE\t", chroma.shape)
    print("MEL\t", mel.shape)
    print("CONTRAST\t", contrast.shape)
    print("TONNETZ\t", tonnetz.shape)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,file_ext="*.wav"):
    features= np.empty((0,193))
    for fn in glob.glob(os.path.join(parent_dir, file_ext)):
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            print(fn)
        except Exception as e:
            print ("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
    return np.array(features)

main_dir = "DATASET\\LibriSpeech\\dev-clean\\84\\121123"  
print ("\ncollecting features and labels...")  
print("\nthis will take some time...")  
features= parse_audio_files(main_dir)  
print("done features")  
np.save('output\\X_84_121123',features)  
