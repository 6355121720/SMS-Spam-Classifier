from idlelib.pyparse import trans

import streamlit as st
import pickle
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt_tab')


ps = PorterStemmer()
tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('modle.pkl','rb'))


def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  text=[i for i in text if i.isalnum()]
  text=[i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
  text=[ps.stem(i) for i in text]
  text=" ".join(text)
  text=tfidf.transform([text])
  return text

st.title('SMS Spam Classfier')

input_sms = st.text_area('Enter SMS')

if st.button('Submit'):
  if model.predict(transform_text(input_sms).reshape(1,-1))==1:
    st.header('Spam')
  else:
    st.header('Not Spam')
