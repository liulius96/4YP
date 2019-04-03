import pickle
from gensim import models, corpora
import nltk
from gensim import corpora,models
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import pandas as pd
import datetime
from text_preprocessing import clean_text
from nltk.corpus import stopwords

# import the dataframe containing all the posts
df = pd.read_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/Initial_DF_all.pkl')


# change the DateTime column from string to datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])
# create a new column for date and time
df['Date'] = [datetime.datetime.date(d) for d in df['DateTime']]
df['Time'] = [datetime.datetime.time(d) for d in df['DateTime']]
# set the index to date
df.index = df.Date
df = df.sort_index()

# create a series of held out documents to be tested - This will be 3 months of documents from Jan 18 - March 18
df_heldout = df.truncate(before=datetime.date(year=2018,month=1,day=1),after=datetime.date(year=2018,month=3,day=31))

# cut the dataframe so that it only contains values from 2014 - 2018
df = df.truncate(before=datetime.date(year=2014,month=1,day=1),after=datetime.date(year=2017,month=12,day=31))
df = df.drop(['DateTime', 'Date', 'Time'], axis=1)

# lemmatize the data
lemmatized_data = []
for post in df.Content:
    lemmatized_data.append(clean_text(post))
print('posts have been lemmatized')

# set up stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['from', 'subject', 're', 'edu', 'use'])
STOPWORDS.extend(['s', 'https', 'www', 'http', 'com', 't'])

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(lemmatized_data)
print(dictionary)
dictionary.save('C:/Users/tliu/Documents/4YP/Outputs/dictionary_2014-2018.dict')

# Buiid the corpus
corpus = [dictionary.doc2bow(text) for text in lemmatized_data]
print('corpus has been built')
print(corpus[0:4])

# save corpus and lemmatized data
with open('C:/Users/tliu/Documents/4YP/Outputs/corpus.pkl', 'wb') as fp:
    pickle.dump(corpus, fp)
    pickle.dump(lemmatized_data, fp)


# LDA model
# Enable logging for gensim - optional`
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

for NUM_TOPICS in range(5,35,5):
    # Build the LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       num_topics=NUM_TOPICS,
                                       id2word=dictionary,
                                       alpha='auto',
                                       random_state=100,
                                       update_every=1,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True
                                       )
    # Save the LDA model
    lda_model.save('C:/Users/tliu/Documents/4YP/Outputs/LDA_arima_model/lda_model_14-18_auto_alpha' + str(NUM_TOPICS) + '.model')
    # prepare the visualisation data
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    # save the visual data as a html file
    pyLDAvis.save_html(vis_data, 'C:/Users/tliu/Documents/4YP/Outputs/pyldavis_14-18_auto_alpha_' + str(
        NUM_TOPICS) + '_topics.html')
