import numpy as np
import pandas as pd
import re
import spacy
from textstat.textstat import textstatistics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import os
import string
import openpyxl
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')


# Break whole text into sentences using spacy
def break_into_sentences(text):
    sen = spacy.load('en_core_web_sm')
    doc_text = sen(text)
    return list(doc_text.sents)


# Avg sentence length
def avg_sentence_length(text):
    regular_punct = list(string.punctuation)

    sentences_list = break_into_sentences(text)
    word_count = 0
    for sentence in sentences_list:
       list2 = [str(token) for token  in sentence]

       for punc_element in list2:
            for punc in regular_punct:
                  if punc == punc_element:
                        list2.remove(punc_element)

       element_to_remove = '\n'
       list2 = [i for i in list2 if i != element_to_remove]

       word_count += len(list2)

    len_sentences = len(break_into_sentences(text))
    average_sentence_length = (word_count / len_sentences)
    return round(average_sentence_length,2)


# Remove punctuation
def remove_punctuation(text):
    regular_punct = list(string.punctuation)
    for punc in regular_punct:
        if punc in text:
            text = text.replace(punc, '')
    punc_text = text.strip()
     
    filtered_text = re.sub(r"http\S+", "", punc_text)
    return filtered_text


# Tokenize
def tokenize(text):
    text_tokenize = word_tokenize(text)
    return text_tokenize


# Word count
def word_count(tokienized_text):
    lent = len(tokienized_text)
    return lent


# Avg word length
def avg_word_length(tokenized_text):
  sum_word_length = 0
  for element in tokenized_text:  
    sum_word_length += len(element)
    avg_length = sum_word_length / len(tokenized_text)
  return round(avg_length,2)


# Stop words
def remove_stop_words(tokenized_text):
    stop_words = set(stopwords.words("english"))
    words = [word for word in tokenized_text if word not in stop_words]
    return words

# Lemmatize
def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return words_lemmatized


# Sentiment analysis
def sentiment_analyzer(words, words_lemmatized):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = [sia.polarity_scores(word)['compound'] for word in words_lemmatized]
    average_sentiment = sum(sentiment_score) / len(sentiment_score)
    positive_words = [word for i, word in enumerate(words) if sentiment_score[i] >= 0.05]
    negative_words = [word for i, word in enumerate(words) if sentiment_score[i] <= -0.05]

    total_words = len(words)
    positive_score = (len(positive_words) / total_words) 
    negative_score = (len(negative_words) / total_words)
    subjectivity_score = (positive_score + negative_score)/ ((total_words) + 0.000001) #Range is from 0 to +1
    return round(average_sentiment,2), round(positive_score,2), round(negative_score,2), '%.5f' % subjectivity_score



# Sylable count
def syllables_count(filter_text):
    return textstatistics().syllable_count(filter_text)
 
def avg_syllables_per_word(text, tokenized_text):
    syllable = syllables_count(text)
    words = len(tokenized_text)
    avg_syllable = syllable / words
    return round(avg_syllable,2)


# Complex words
def difficult_words(tokenized_text):
     
    nlp = spacy.load('en_core_web_sm')
    # doc = nlp(text)
 
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()
     
    for word in tokenized_text:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set), round(((len(diff_words_set)/ len(tokenized_text))*100),2)


# Pronouns
def get_pronouns(text):
    find_pronouns = re.compile(r'\b(I|you|he|she|it|we|they|them|him|her|his|hers|its|theirs|our|your(?-i:us))\b',re.I)
    pronouns = find_pronouns.findall(text)

    final_pronouns = ""
    for element in pronouns:
        if element not in final_pronouns:
            final_pronouns = final_pronouns + "/" + element
            
    return final_pronouns


# Driver Code ................................................................................................

input_url_df = pd.read_csv('input.csv')
raw_df = input_url_df[['URL_ID']]

path = 'output_data/'
raw_temp = raw_df

for file in os.listdir(path):
    print(f'File under progress: {file} ...')
    
    url_id = float(file.split('.txt')[0])
    
    try: 
        with open(os.path.join(path,file), "r", encoding='utf-8') as info_file:
            reader = info_file.read()
        
        pattern = r'[0-9]'
        text = re.sub(pattern,'', reader).lower()
        # //////////////////////////////////////////////////////////////////////////////////////
        break_into_sentences(text)
        avg_sent_length = avg_sentence_length(text)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Avg_sentence_length'] = avg_sent_length
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Avg_number_of_words_per_sentence'] = avg_sent_length
        # //////////////////////////////////////////////////////////////////////////////////////
        filtered_text = remove_punctuation(text)
        tokenized_text = tokenize(filtered_text)
        word_counts = word_count(tokenized_text)
        
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Word_count'] = int(word_counts)
        # //////////////////////////////////////////////////////////////////////////////////////
        avg_word_len = avg_word_length(tokenized_text)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Avg_word_length'] = int(avg_word_len)
        # //////////////////////////////////////////////////////////////////////////////////////
        words = remove_stop_words(tokenized_text)
        words_lemmatized = lemmatize(words)
        polar, pos, neg, sub = sentiment_analyzer(words, words_lemmatized)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Positive_score'] = pos
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Negative_score'] = neg
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Polarity_score'] = polar
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Subjectivity_score'] = sub
        # //////////////////////////////////////////////////////////////////////////////////////
        syllables_count(filtered_text)
        avg_syllable_per_word = avg_syllables_per_word(filtered_text, tokenized_text)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Avg_syllables_per_word'] = avg_syllable_per_word
        # //////////////////////////////////////////////////////////////////////////////////////
        complex_count, complex_word_percent = difficult_words(tokenized_text)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Complex_word_count'] = int(complex_count)
        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Complex_word_percentage'] = complex_word_percent
        # //////////////////////////////////////////////////////////////////////////////////////
        pronouns = get_pronouns(text) 

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Personal_pronouns'] = pronouns
        # //////////////////////////////////////////////////////////////////////////////////////
        fog_index = 0.4*(avg_sent_length + complex_word_percent)

        raw_temp.loc[(raw_temp['URL_ID']==url_id), 'Fog_index'] = fog_index

    except Exception as e :
            print(str(e))   


# Merge both input df and raw_temp
output_df = pd.merge(input_url_df, raw_temp, on = 'URL_ID', how = 'outer')


# for those two url's whose webpage doesn't exit ...values are NaN .. have to handle those
output_df.fillna('Not available', inplace = True)


# Convert output_df to excel
output_df.to_excel(r'F:/My_Disk/python/Projects/NLP/outptut.xlsx', index=False)

print('Processing done!')
