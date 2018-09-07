# -*- coding: utf-8 -*-

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
import matplotlib.pyplot as plt
import unicodedata
import numpy as np
import codecs
import gensim
from gensim.models import Word2Vec
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
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
    s = s.lower().replace('-', '').replace(')', '').replace('(', '').replace('\'', '').replace('\"', '').replace('|', '').replace(';', '').replace(':', '').replace(u'á', 'a').replace('?', '')
    s = s.replace(u'é', 'e').replace(u'í', 'i').replace(u'ó', 'o').replace(u'ú', 'u').replace(u'ñ', 'n').replace(u'\u201d', '').replace(u'\u201c', '').replace(u'\xa0','')
    # lemmatization does not work very well for Spanish so we will use normal tokens
    tokens = []
    tokens = [t for t in s.split(' ')] # tokenization just at blank space
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove digits
    return tokens

# Data processing

df = pd.read_csv('58ea8168-bb5d-4740-b9f5-850024d82aa5_intents.csv', encoding='utf-8')
data = df['pedir actualizacion'].tolist()
text = [' '.join(my_tokenizer(s)) for s in data]
labels = [x+'='+y for y,x in zip(df['pedir actualizacion'].tolist(), df['atencion_actualizar_datos'].tolist())]


'''
tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()
print (X.shape)
dist_array = pdist(X)

# calculate hierarchy
Z = linkage(dist_array, 'ward')


# Check the Cophenetic Correlation Coefficient of your clustering with help of the cophenet() function.
# This (very very briefly) compares (correlates) the actual pairwise distances of all your samples to
# those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering
# preserves the original distances
ccc, coph_dists = cophenet(Z, dist_array)
print (ccc)

print (dist_array.shape)

plt.figure(figsize=(60, 150))
plt.title("Ward")
dendrogram(Z, labels=labels, orientation='left', leaf_font_size=2)
plt.savefig('fase1_ward.png', format='png', dpi=200)
'''


# Try embeddings

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
all_tokens = [my_tokenizer(s) for s in data]
print (len(all_tokens))
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_tokens)]

model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

X2 = np.empty((len(all_tokens), len(all_tokens)))
for i in range(0, len(all_tokens)):
    for j in range(0, len(all_tokens)):
        X2[i,j] = 1 - model.docvecs.similarity(i,j)

print (X2)
print (X2.shape)

# calculate hierarchy
h, w = X2.shape
Z = linkage(X2[np.triu_indices(h, 1), 'ward'])
#Z = linkage(X2, 'ward')

# Check the Cophenetic Correlation Coefficient of your clustering with help of the cophenet() function.
# This (very very briefly) compares (correlates) the actual pairwise distances of all your samples to
# those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering
# preserves the original distances
try:
    ccc, coph_dists = cophenet(Z, dist_array)
    print (ccc)
except:
    pass

plt.figure(figsize=(60, 150))
plt.title("Ward")
dendrogram(Z, labels=labels, orientation='left', leaf_font_size=2)
plt.savefig('fase1_ward_emb_triagle.png', format='png', dpi=200)

# Actually the perform worst that tf-idf



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
