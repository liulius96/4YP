import pickle
import pandas as pd
import datetime
import numpy as np
import json
import nltk
import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
import logging
from gensim import corpora,models
import re
import numpy as np
import matplotlib.pyplot as plt
from ptm.nltk_corpus import get_ids_cnt
from ptm.utils import convert_cnt_to_list, get_top_words
from sklearn.feature_extraction.text import CountVectorizer
import os

STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['from', 'subject', 're', 'edu', 'use'])
STOPWORDS.extend(['s', 'https', 'www', 'http', 'com', 't'])

df = pd.read_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/merged_df_slda.pkl')


# initialise the word vectoriser with stop words and the specficied dtype
vectoriser = CountVectorizer(stop_words=STOPWORDS, dtype=np.int32)

# create a bag of words by vectorising every post in df.post
bow = vectoriser.fit_transform(df.post)

# change the bag of words into an ndarray
bow = bow.toarray()
bow = bow.T

# initialise a series for regression
labels = np.asarray(df.log_ret.values, dtype=np.int32)

with open('C:/Users/tliu/Documents/4YP/sLDA/data/train_data_slda.npy', 'wb') as d:
    np.save(d,bow)
    np.save(d,labels)

#
#
#   Run the code in ubuntu (LDA++)
# fslda online_train /mnt/c/Users/tliu/Documents/4YP/sLDA/data/train_slda.npy /mnt/c/Users/tliu/Documents/4YP/sLDA/data/slda_model.npy --topics 20
#
#

with open('C:/Users/tliu/Documents/4YP/sLDA/data/slda_model_initial_test.npy', 'rb') as model:
    alpha = np.load(model)
    beta = np.load(model)
    eta = np.load(model)

# Print the contents of alpha. We have trained our model with 20 topics and
# subsequently the shape of alpha is (20, 1)
alpha

# Print the contents of beta. The shape of beta is (20, 63950), as it refers to
# the per topic word distributions. This is a probability of every word occurring in every topic.
beta

# print the contents of eta
eta


#
#
# Now, we use the trained model to transform our data with the following bash command.
# fslda transform -q /mnt/c/Users/tliu/Documents/4YP/sLDA/data/slda_model_initial_test.npy /mnt/c/Users/tliu/Documents/4YP/sLDA/data/train_data_slda.npy /mnt/c/Users/tliu/Documents/4YP/sLDA/data/initial_test_transformed.npy
#
#

with open('C:/Users/tliu/Documents/4YP/sLDA/data/initial_test_transformed.npy', "rb") as f:
    Z = np.load(f)

# Print the contents of Z, which is the per topic document distribution. We
# could say that Z[0] is the number of words that were produced from topic 0.
Z