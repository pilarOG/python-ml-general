# -*- coding: utf-8 -*-

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import unicodedata
import numpy as np
import codecs
import gensim

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

# Some spanish stopwords (with some misspellings too) and some specific for this data
stopwords = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los', u't', u'd',
         u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el', u'le', u'p', u'hay',
         u'esta', u'lo', u'fue', u'es', u'quien', u'su', u'sus', u'mas', u'durante', u'hasta', u'estos',
         u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a', u'ha', u'luego', u'estar',
         u'respectivamente', u'asimismo', u'l', u'les', u'montt', u'nos', u'va', u'emol', u'ademas',
         u'son', u'ese', u'era', u'eran', u'ser', u'm', u'e', u'g', u'esos', u'eso', u'asi', u'esa', u'esto',
         u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque', u'este', u'']

# Function to very naive tokenization of words for Spanish
# Input is a string/sentence, output a list ok tokens
def my_tokenizer(s):
    # Replace symbols and vowels with diacritics (and other leftovers from the newspaper website)
    s = s.lower().replace('-', '').replace(')', '').replace('(', '').replace('\'', '').replace('\"', '').replace('|', '').replace(';', '').replace(':', '').replace(u'á', 'a')
    s = s.replace(u'é', 'e').replace(u'í', 'i').replace(u'ó', 'o').replace(u'ú', 'u').replace(u'ñ', 'n').replace(u'\u201d', '').replace(u'\u201c', '').replace(u'\xa0','')
    # lemmatization does not work very well for Spanish so we will use normal tokens
    tokens = []
    tokens = [t for t in s.split(' ')] # tokenization just at blank space
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove digits
    return tokens

df = pd.read_csv('58ea8168-bb5d-4740-b9f5-850024d82aa5_intents.csv', encoding='utf-8')
text = df['pedir actualizacion'].tolist()
labels = df['atencion_actualizar_datos'].tolist()

tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()

dist_array = pdist(X)

# calculate hierarchy
Z = linkage(dist_array, 'ward')

plt.figure(figsize=(150, 60))
plt.title("Ward")
dendrogram(Z, labels=labels)
plt.savefig('fase1.svg', format='svg', dpi=1200)





'''
df = pd.read_csv('all_data_v1.csv', encoding='utf-8')
text = df['estado de la devolucion'].tolist()
labels = df['credito_acreencias'].tolist()

tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()

dist_array = pdist(X)

# calculate hierarchy
Z = linkage(dist_array, 'ward')

plt.figure(figsize=(150, 60))
plt.title("Ward")
dendrogram(Z, labels=labels)
plt.savefig('fase2.svg', format='svg', dpi=1200)
'''
