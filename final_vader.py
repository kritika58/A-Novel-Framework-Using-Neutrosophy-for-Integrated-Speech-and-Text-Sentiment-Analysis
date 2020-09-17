from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import digits
import csv
import glob
import os
main_dir = "C:\\Users\\lenovo\\Desktop\\FINAL PROJECT\\LibriSpeech_train\\train-clean-100\\"
sub_dirs=os.listdir(main_dir)  
print(sub_dirs)
file_ext="*.txt"

with open('vader_train_new.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for sub_dir in enumerate(sub_dirs):
        print("sub_dir:", sub_dir[1])
        sub_dirs_2=os.listdir(str(main_dir+sub_dir[1]))
        for sd in enumerate(sub_dirs_2):
            for fn in glob.glob(os.path.join(main_dir, sub_dir[1], sd[1], file_ext)):
                print(fn)
                file1=open(str(fn))
                Lines = file1.readlines() 
                # Strips the newline character 
                for line in Lines: 
                    remove_digits = str.maketrans('', '', digits)
                    sentence = line.translate(remove_digits)
                    sid_obj = SentimentIntensityAnalyzer()
                    sentiment_dict = sid_obj.polarity_scores(sentence) 
                    #print(sentence)
                    s=0
                    if sentiment_dict['compound'] >= 0.05 : 
                        #"Positive" 
                        s=0
                    elif sentiment_dict['compound'] <= - 0.05 : 
                        #"Negative" 
                        s=2
                    else : 
                        #"Neutral"
                        s=1
                    writer.writerow([sentiment_dict['pos'], sentiment_dict['neu'], sentiment_dict['neg'], sentiment_dict['compound'], s])