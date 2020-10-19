# A Novel Framework Using Neutrosophy for Integrated Speech and Text Sentiment Analysis

## Link To The Paper
https://www.mdpi.com/2073-8994/12/10/1715/htm

## Abstract
With increasing data on the Internet, it is becoming difficult to analyze every bit and make sure it can be used efficiently for all the businesses. One useful technique using Natural Language Processing (NLP) is sentiment analysis. Various algorithms can be used to classify textual data based on various scales ranging from just positive-negative, positive-neutral-negative to a wide spectrum of emotions. While a lot of work has been done on text, only a lesser amount of research has been done on audio datasets. An audio file contains more features that can be extracted from its amplitude and frequency than a plain text file. The neutrosophic set is symmetric in nature, and similarly refined neutrosophic set that has the refined indeterminacies I1 and I2 in the middle between the extremes Truth T and False F. Neutrosophy which deals with the concept of indeterminacy is another not so explored topic in NLP. Though neutrosophy has been used in sentiment analysis of textual data, it has not been used in speech sentiment analysis. We have proposed a novel framework that performs sentiment analysis on audio files by calculating their Single-Valued Neutrosophic Sets (SVNS) and clustering them into positive-neutral-negative and combines these results with those obtained by performing sentiment analysis on the text files of those audio.

#### Task 1: Topic Selection and Literature Survey
Having worked on NLP projects earlier, I was sure that I would like to choose a topic from this domain and work on it for my final year project. Sentiment analysis is a very topic with a wide range of applications in the present day. Hence I included the concept of neutrosophy coupling it with speech analysis because the trend of voice integration has resulted in a surplus amount of data inform of audio files.

#### Task 2: Selecting a dataset
Dataset played a crucial role in this project. The reason being I wanted to map audio SVNS to text SVNS for comparison so a dataset with audio translation was required. Hence LibriSpeech dataset was picked. For the demonstration of this project the following two folders were used: Dev-clean (337 MB) and Train-clean-100 (6.3 GB)
    
#### Task 3: Processing audios from .flac to .wav
The dataset was available in .flac format. It was necessary to convert these files into .wav format for further processing and extracting features. For this ffmpeg was used in shell script with bash. FFmpeg is a free and open-source project consisting of a vast software suite of libraries and programs for handling video, audio, and other multimedia files and streams.

#### Task 4: Extracting Features and Preprocessing
The audio files were then fed into the python feature extraction script which extracted 193 features per audio file. The following npy files were generated as result: X_dev.npy (2703 x 193) and X_train.npy (28539 x 193). Then these files were normalized using sklearn. 

#### Task 5: Generating audio SVNS
Since the dataset was unlabeled, K means algorithm was used for clustering. With K being set to 3, the clusters were obtained. Let the cluster centres be A, B and C. For every data point P in the dataset distance was calculated to the centres of each cluster. 1-distance implied the closeness measure to each cluster or class (positive, neutral or negative). This is how SVNS were created and stored in a csv file.

#### Task 6: Sentiment Analysis of text translation using VADER
VADER is a tool used for sentiment analysis which provides a measure for positive, neutral and negative classes for each input sentence. Using VADER text translation for each audio was analyzed and SVNS were generated. 

#### Task 7: Clustering and visualizing text SVNS
Taking the csv file of text SVNS as input, K means cluster with K being as 3 was performed.

#### Task 8: Combining the SVNS
Audio SVNS- <PA, IA, NA>; Text SVNS- <PT, IT, NT>; Combined SVNS- <PC, IC, NC>; where, PC = (PT + PA)/2, IC = (IT + IA)/2 and NC = (NT + NA)/2. This is how the SVNS were combined.

#### Task 9: Visualization of combined SVNS
Using K means clustering and hierarchical agglomerative clustering algorithms, the SVNS were visualized into 3 clusters.

#### Task 10: Results and Documentation
At last the documentation of the entire project was put together in the required format as per the university guidelines.
