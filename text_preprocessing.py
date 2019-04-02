import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
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

#
# Define the functions we will use
#
# Most of the information came from the link below
# https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html?fbclid=IwAR393vGJvaKKZIV_s-TGKYJgrh3QKdlnG2_gEZleCz4wTpUOvj-PZ4B-Pd8
#

STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['from', 'subject', 're', 'edu', 'use'])
STOPWORDS.extend(['s', 'https', 'www', 'http', 'com', 't'])

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        try:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        except inflect.NumOutOfRangeError:
            new_words.append(word)
    return new_words


def remove_stopwords(STOPWORDS, words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in STOPWORDS:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def clean_text(text):
    """This is the overall function to organise the text"""
    text = remove_between_square_brackets(text)
    text = replace_contractions(text)
    text = word_tokenize(text.lower())      # the text is tokenized at this line
    # text = text.lower()
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    # tokenized_text = replace_numbers(tokenized_text) # This was commented out because it was raising errors
    text = [t for t in text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    # text = remove_stopwords(STOPWORDS, text)
    text = lemmatize_verbs(text)
    return text

