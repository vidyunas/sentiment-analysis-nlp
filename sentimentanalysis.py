import pandas
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


// IMPORTING CSV

filepath = r"C:\Users\T02574\Documents\senti.csv"
df = pandas.read_csv(filepath,encoding="ISO-8859-1")

//DROPPING MISSING VALUES

df=df.dropna()

//GETTING SERIES FROM DATAFRAME

df1=df["Comments"]

//CONVERTING SERIES TO ARRAY

a=pandas.Series(df1).values

//OBJECT CREATION TO USE 

obj = SentimentIntensityAnalyzer()

//STORING THE SCORE IN LIST
//EMPTY LIST
lpos=[]
lneg=[]
lneu=[]
lcom=[]

for i in a:
    sentiment_dict = obj.polarity_scores(i)
    lneg.append(sentiment_dict['neg'])
    lpos.append(sentiment_dict['pos'])
    lneu.append(sentiment_dict['neu'])
    lcom.append(sentiment_dict['compound'])

//CONVERTING ARRAY TO SERIES TO FIT IN DATAFRAME

spos=pandas.Series(lpos)
sneg=pandas.Series(lneg)
sneu=pandas.Series(lneu)
scom=pandas.Series(lcom)

//ADDING NEW COLUMNS IN DATAFRAME

df['negativity']=sneg
df['positivity']=spos
df['neutrality']=sneu
df['compond']=scom

//FINDING FINAL SENTIMENT USING COMPOUND OF THE SENTENCE AND STORING IT IN LIST

final_sentiment=[]
for i in range(0,len(lcom)):    
    if(lcom[i] >= 0.05): 
        final_sentiment.append("Positive") 
  
    elif(lcom[i] <= - 0.05) : 
        final_sentiment.append("Negative") 
  
    else : 
        final_sentiment.append("Neutral")
//ADDING NEW COLUMN RESULT FOR CONCLUSION 

sentiment=pandas.Series(final_sentiment)
df['result']=sentiment

//EXPORTING DATAFRAME TO CSV FORMAT

export_csv = df.to_csv (r'C:\Users\T02574\Documents\final.csv', index = None, header=True)
