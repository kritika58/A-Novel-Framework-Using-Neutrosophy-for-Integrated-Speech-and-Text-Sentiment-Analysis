from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import digits
import csv
# Create a SentimentIntensityAnalyzer object

file1 = open("DATASET\\LibriSpeech\\dev-clean\\84\\121123\\84-121123.trans.txt", 'r') 
Lines = file1.readlines() 

with open('vader_84_121123.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Strips the newline character 
    for line in Lines: 
        remove_digits = str.maketrans('', '', digits)
        sentence = line.translate(remove_digits)
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sentence) 
        print(sentence)
        s=0
        if sentiment_dict['compound'] >= 0.05 : 
            #Positive 
            s=0
        elif sentiment_dict['compound'] <= - 0.05 : 
            #Negative 
            s=2
        else : 
            #Neutral
            s=1
        writer.writerow([sentiment_dict['pos'], sentiment_dict['neu'], sentiment_dict['neg'], sentiment_dict['compound'], s])
