from gensim import corpora,models
import os
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
from ptm import GibbsSupervisedLDA
from ptm.nltk_corpus import get_ids_cnt
from ptm.utils import convert_cnt_to_list, get_top_words
import json

logger = logging.getLogger('GibbsSupervisedLDA')
logger.propagate = False

# load the prepared LDA model, corpus and dictionary
lda_model = models.LdaModel.load('C:/Users/tliu/Documents/4YP/Outputs/lda_model_10.model')

dictionary = open('C:/Users/tliu/Documents/4YP/Outputs/dictionary_all.dict')

with open('C:/Users/tliu/Documents/4YP/Outputs/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

path_to_json = 'C:/Users/tliu/Documents/4YP/bitcointalk_replies_raw'


