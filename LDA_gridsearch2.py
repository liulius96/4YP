from nltk.corpus import stopwords
from gensim import models, corpora
import nltk
from gensim import corpora,models
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import json
import os
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import pickle
import datetime
from text_preprocessing import clean_text


dictionary = open('C:/Users/tliu/Documents/4YP/Outputs/dictionary_all.dict')

with open('C:/Users/tliu/Documents/4YP/Outputs/corpus.pkl', 'rb') as f:
    corpus, lemmatized_data = pickle.load(f)