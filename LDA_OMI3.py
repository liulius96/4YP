# this is exactly the same as LDA_OMI1 except we filter out the extreme terms
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
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


NUM_TOPICS = 5

# Enable logging for gensim - optional`
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

perplexity = {}
coherence = {}


STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['from', 'subject', 're', 'edu', 'use'])
STOPWORDS.extend(['s', 'https', 'www', 'http', 'com', 't'])
# STOPWORDS.extend(['Bitcoin', 'bitcoin', 'btc'])

# list of unique documents
# seen = []

# list of tuples with documents
arr = []

path_to_json = 'C:/Users/tliu/Documents/4YP/bitcointalk_replies_raw'

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

for file in json_files:
    with open(path_to_json + '/' + file, encoding="utf8") as json_data:
        print(str(file))
        datum = json.load(json_data)
        for post in range(0, len(datum['posts'])):
            topic_dt = datum['posts'][post]['date']
            topic_content = datum['posts'][post]['content']
            # if topic_content not in seen:
            #     seen.append(topic_content)
            arr.append((topic_dt,topic_content))

            for reply in range(0, len(datum['posts'][post]['replies'])):
                reply_content = datum['posts'][post]['replies'][reply]['content']
                reply_dt = datum['posts'][post]['replies'][reply]['date']
                # if reply_content not in seen:
                #     seen.append(reply_content)
                arr.append((reply_dt, reply_content))


print('Number of Documents: ' + str(len(arr)))

newarr = []
seen = set()
for (a, b) in arr:
    if b not in seen:
        newarr.append((a, b))
        seen.add(b)
print('Number of Documents (no repeats): ' + str(len(newarr)))


# Initialise a dataframe with dates
df = pd.DataFrame(newarr, columns=['DateTime', 'Content'])
df.to_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/Initial_DF_all.pkl')

#  sort the information
# df['DateTime'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in df['DateTime']]
# df['Date'] = [datetime.datetime.date(d) for d in df['DateTime']]
# df['Time'] = [datetime.datetime.time(d) for d in df['DateTime']]


posts = [i[1] for i in newarr]
tokenized_data =[]
for text in posts:
    tokenized_data.append(clean_text(text))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)


# Filter out the extremes
dictionary.filter_extremes(no_below=50)
print(dictionary)

# Save the dictionary
dictionary.save('C:/Users/tliu/Documents/4YP/Outputs/dictionary_all.dict')

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# Save the corpus
with open('C:/Users/tliu/Documents/4YP/Outputs/corpus.pkl', 'wb') as fp:
    pickle.dump(corpus, fp)
    pickle.dump(tokenized_data, fp)


# Have a look at how the 20th document looks like: [(word_id, count), ...]
print(corpus[20])



# Build the LDA model
lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary,alpha='auto')

# Save the LDA model
lda_model.save('C:/Users/tliu/Documents/4YP/Outputs/lda_model_auto_alpha'+str(NUM_TOPICS)+'.model')

# prepare the visualisation data
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# save the visual data as a html file
pyLDAvis.save_html(vis_data, 'C:/Users/tliu/Documents/4YP/Outputs/output_filter_extremes_auto_alpha_' + str(NUM_TOPICS) + '_topics.html')

# get the topics for the entire corpus
topics = lda_model.get_document_topics(corpus, per_word_topics=True)

# store all the topic data for each document in a variable
all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]

# create a new vector to hold all the most likely topics for each relevant document
arr_topic = []

# run through and store the most likely topic for each document
for doc in all_topics:
    tuples = doc[0]
    max_tuple = max(doc[0], key=lambda item: item[1])
    arr_topic.append(max_tuple[0])

df2 = pd.DataFrame({
    'DateTime': df['DateTime'],
    'Document Content': df['Content'],
    'Most Likely Topic': arr_topic
})

# save the df
df2.to_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/most_likely_topic_dataframe_low_alpha_' + str(NUM_TOPICS)+'_topics.pkl')

#
#     # Compute Perplexity
#     print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
#
#     # Add perplexity score to dictionary
#     perplexity['Number of topics = ' + str(x)] = 'Perplexity Score: ' + str(lda_model.log_perplexity(corpus))
#
#     # Compute Coherence Score
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)
#
#     # Add Coherence Score to dictionary
#     coherence['Number of topics = ' + str(x)] = 'Coherence Score: ' + str(coherence_lda)
#
#
# path_pickle = 'C:/Users/tliu/PycharmProjects/4YP'
#
# # pickle the perplexity and coherence dictionary
# file_p = open(path_pickle + '/perplexity.pkl', 'wb')
# pickle.dump(perplexity, file_p)
#
# # pickle the coherence dictionary
# file_c = open(path_pickle + '/coherence.pkl', 'wb')
# pickle.dump(coherence, file_c)