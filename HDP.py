import pickle
from gensim import models, corpora
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import pandas as pd
import datetime
from gensim.models import HdpModel

# Enable logging for gensim - optional`
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# load the dictionary
dictionary = gensim.corpora.Dictionary.load('C:/Users/Tony/Documents/4YP/data/pickles/dictionary_2014-2018.dict')
dictionary.filter_extremes(no_below=100)

# load the corpus and lemmatized data
with open('C:/Users/Tony/Documents/4YP/data/pickles/corpus14-18.pkl', 'rb') as fp:
    corpus = pickle.load(fp)
    lemmatized_data=pickle.load(fp)

# Build the hdp model
hdp = HdpModel(corpus, dictionary)
topics = hdp.get_topics()
print('Number of topics is '+str(len(topics)))