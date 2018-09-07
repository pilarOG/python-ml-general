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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from sklearn.feature_extraction.text import TfidfVectorizer

# Some spanish stopwords (with some misspellings too) and some specific for this data
stopwords = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los', u't', u'd', u'te',
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



########## MAIN #################
# Best combination of hyperparameters found: 300-6-3-40-30
def main(embedding_vector_size=100,
         embedding_window_size=5,
         embedding_min_count=3):

    # Load the data and split at some naive sentence boundaries
    # I'm doing this to have more documents given that I could find just 30 to 40 news of uneven lengths
    cleaned_docs = []
    docs = codecs.open('noticias-emol.txt', encoding='utf-8').read().split('\n')
    [cleaned_docs.append(do) for do in docs if 'http' not in do and do] # This is an extra step to take out the links in the data
    docs = [line.rstrip() for line in '\n'.join(cleaned_docs).replace(',', '\n').replace('.', '\n').replace('\"', '\n').split('\n')]
    print ('Number of document fragments: '+str(len(docs)))

    # We tokenize each word in the fragments, turning each into a list of tokens
    all_tokens = []
    [all_tokens.append(my_tokenizer(doc)) for doc in docs]

    # Word embedding training: a list of list is feed into the model
    PYTHONHASHSEED=1 # This is to ensure reproducible results when training the word2vec, so the analysis in my readme makes some sense
    model = Word2Vec(size=embedding_vector_size, window=embedding_window_size, min_count=embedding_min_count) # Hyperparameters to experiment with
    model.build_vocab(all_tokens)
    model.train(all_tokens, total_examples=model.corpus_count, epochs=model.iter)
    word_vectors = model.wv

    # Given the trained embeddings we can already get some interesting results. Here we are searching
    # in the model some specific relevant word given the topic and the function allows us to get back the
    # other words that are closest to the given word. In the Readme there is a deeper analysis of this.
    #print("Words closest to \"hombre\"")
    #print(word_vectors.similar_by_word("hombre"))
    #print("Words closest to \"mujer\"")
    #print(word_vectors.similar_by_word("mujer"))
    #print("Words closest to \"asesino\"")
    #print(word_vectors.similar_by_word("asesino"))
    #print("Words closest to \"victima\"")
    #print(word_vectors.similar_by_word("victima"))

    # Given the trained we will build a matrix of similarity between all words,
    # of sixe N_words X N_words, as we want to cluster similar words.
    # To build it using gensim directly I used this code, you can also check the documentation
    # for more detail
    # https://groups.google.com/forum/#!topic/gensim/gfOuXGzvsA8
    # https://radimrehurek.com/gensim/models/keyedvectors.html
    similarity_matrix = []
    index = gensim.similarities.MatrixSimilarity(gensim.matutils.Dense2Corpus(model.wv.syn0.T))
    [similarity_matrix.append(sims) for sims in index]

    similarity_array = np.array(similarity_matrix)
    word_index = model.wv.index2word

    Z = linkage(similarity_array, 'single', metric=None)
    # First mergin
    print ('First two words merged: ', word_index[int(Z[0][0])], ' ', word_index[int(Z[0][1])])



    # Check the Cophenetic Correlation Coefficient of your clustering with help of the cophenet() function.
    # This (very very briefly) compares (correlates) the actual pairwise distances of all your samples to
    # those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering
    # preserves the original distances
    #ccc, coph_dists = cophenet(Z, similarity_array)
    #print (ccc)

    # Plot dendogram
    plt.figure(figsize=(60, 150))
    plt.title("Hierarchical Clustering Dendrogram with Ward Linkage for word similarity")
    plt.xlabel('distance')
    plt.ylabel('sample index')
    dendrogram(Z, labels=word_index, orientation='left', leaf_font_size=5)
    plt.savefig('femicides_ward.png', format='png', dpi=200)


# Run predfined parameters
main()
