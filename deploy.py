#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import string # special operations on strings
import spacy # language models
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import STOPWORDS
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk import pos_tag
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import base64
import textblob
from textblob import TextBlob
import pickle
from pickle import load
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

st.title('HOTEL REVIEW ANALYSIS')
st.subheader('Using NLP')

if st.checkbox('Show Sentiment Analysis'):
    st.subheader('Sentiment of your Review')
    message = st.text_area("REVIEW", """""",key = 'text')
    if st.button('Analyze'):
        blob=TextBlob(message)
        result_sentiment=blob.sentiment
        st.success(result_sentiment)
        
svm_deploy=open('F:\pickles\svm.pkl','rb')
model=pickle.load(svm_deploy)

cv_deploy=open(r'F:\pickles\tf_vector.pkl','rb')
vector=pickle.load(cv_deploy)

lemmatizer = WordNetLemmatizer()
w_tokenizer=WhitespaceTokenizer()

sw=set(stopwords.words('english'))


def text_preprocessing(quo):
  quo=re.sub('[^a-zA-Z]',' ',quo)
  quo=quo.lower().split()
  clean_word=[i for i in quo if not i in sw]
  sen=' '.join(clean_word)
  return sen
cleaned_text=text_preprocessing('message')

def lemmatize(txt):
  list_review=[lemmatizer.lemmatize(word=word, pos=tag[0].lower()) if tag[0].lower() in ['a','r','n','v'] else word for word, tag in pos_tag(word_tokenize(txt))]
  return (' '.join([x for x in list_review if x]))

line=[lemmatize(cleaned_text)]
z=vector.transform(line)
  
if st.button("Emotion"):

    wordnet = WordNetLemmatizer()
    text = re.sub('[^A-za-z0-9]', ' ', message)
    text = text.lower()
    text = text.split(' ')
    text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
    text = ' '.join(text)

    svm_deploy = open('F:\pickles\svm.pkl', 'rb')
    model = pickle.load(svm_deploy)

    cv_deploy = open(r'F:\pickles\tf_vector.pkl', 'rb')
    vector = pickle.load(cv_deploy)

    transformed_input = vector.transform([text])

    if model.predict(transformed_input) == 0:
        st.error("This is a Negative Review 😟!")

    elif model.predict(transformed_input) == 1:
        st.warning("This is a Neutral Review 🙂!")
        st.snow()
    else:
        st.success("This is a Positive Review 😍!")

        st.balloons()




