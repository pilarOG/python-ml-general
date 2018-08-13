# -*- coding: utf-8 -*-

# Most of this code belongs to:
# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

############################################################################
# The soft k-means algorithm is commented through out the code to understand the lectures. I'll try to add a read me with the formulas to have the full explanation!
#
# For other explanations:
# http://rosalind.info/problems/ba8d/
# https://blog.tshw.de/wp-content/uploads/2009/03/soft-clustering.pdf
#
# Symbols:
#
# D = dimensionality, number of features
# N = number of samples
# X = N x D matrix, input data to the algorithm
# K = number of clusters
# M = K x D matrix of means or cluster centers
# R = K x N responsability matrix, how much each sample belongs to each k
# beta = stiffnes parameter, or amount of flexibility in soft assignment
# d = distance
#
############################################################################


from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
import codecs
import unicodedata

# Some spanish stopwords (with some misspellings too)
spanish_stopwords = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
         u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
         u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a', u'ha',
         u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque', u'este', u'']

# Function to very naive tokenization of words for Spanish
# Input is a string/sentence, output a list ok tokens
def my_tokenizer(s):
    # Replace symbols and vowels with diacritics (and other leftovers from the newspaper website)
    s = s.lower().replace('-', '').replace(')', '').replace('(', '').replace('\"', '').replace(';', '').replace(':', '').replace(u'á', 'a')
    s = s.replace(u'é', 'e').replace(u'í', 'i').replace(u'ó', 'o').replace(u'ú', 'u').replace(u'ñ', 'n').replace(u'\u201d', '').replace(u'\u201c', '').replace(u'\xa0','')
    # lemmatization does not work very well for Spanish so we will use normal tokens
    tokens = []
    tokens = [t for t in s.split(' ')] # tokenization just at blank space
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove digits
    return tokens

# Function to create one-hot vector tokens for each word
# Each word will consist on a vector of 1 and 0s with the length of the documents
# meaning that the vector is saying if the word appears in the document or not
# TODO: confirm this
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    return x

# Function to measure distance
def d(u, v):
    diff = u - v
    return diff.dot(diff)

# Function to calculate cost
def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

# Main algorithm, soft k-means
def soft_k_means(X, K, index_word_map, max_iter=20, beta=1.0, show_plots=True):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    exponents = np.empty((N, K))

    # initialize M to random
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter)
    for i in range(max_iter):
        # step 1: determine assignments / resposibilities
        # is this inefficient?
        for k in range(K):
            for n in range(N):
                # R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum( np.exp(-beta*d(M[j], X[n])) for j in range(K) )
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))

        R = exponents / exponents.sum(axis=1, keepdims=True)

        # step 2: recalculate means
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()

        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 10e-5:
                break
        print ('cost', costs[i])

    if show_plots:
        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.figure(figsize=(80.0, 80.0))
        plt.scatter(X[:,0], X[:,1], s=300, alpha=0.9, c=colors)
        annotate1(X, index_word_map)
        # plt.show()
        plt.savefig("test.png")


    # print out the clusters
    hard_responsibilities = np.argmax(R, axis=1) # is an N-size array of cluster identities
    # let's "reverse" the order so it's cluster identity -> word index
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
      word = index_word_map[i]
      cluster = hard_responsibilities[i]
      if cluster not in cluster2word:
        cluster2word[cluster] = []
      cluster2word[cluster].append(word)

    # print out the words grouped by cluster
    for cluster, wordlist in cluster2word.items():
      print("cluster", cluster, "->", wordlist)

    return M, R, costs, X

# Function to annotate scatter plot
def annotate1(X, index_word_map, eps=0.1):
  N, D = X.shape
  placed = np.empty((N, D))
  for i in range(N):
    x, y = X[i]

    # if x, y is too close to something already plotted, move it
    close = []

    x, y = X[i]
    for retry in range(3):
      for j in range(i):
        diff = np.array([x, y]) - placed[j]

        # if something is close, append it to the close list
        if diff.dot(diff) < eps:
          close.append(placed[j])

      if close:
        # then the close list is not empty
        x += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        y += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        close = [] # so we can start again with an empty list
      else:
        # nothing close, let's break
        break

    placed[i] = (x, y)
    #print (index_word_map[i])
    plt.annotate(
      s=index_word_map[i],
      xy=(X[i,0], X[i,1]),
      xytext=(x, y),
      arrowprops={
        'arrowstyle' : '->',
        'color' : 'black',
      }
    )


########## MAIN #################

# Load the data and split at some naive sentence boundaries
# I'm doing this to have more documents given that I could find just 10 news
docs = [line.rstrip() for line in codecs.open('noticias-tercera.txt', encoding='utf-8').read().replace(',', '\n').replace('.', '\n').split('\n')]
stopwords = set(spanish_stopwords)

print ('Number of documents')
print (len(docs))

# We will test two representations of the input: word2vec embeddings and a one-hot index of the words
# For the second one, we will take out all words with few frequence (< 3) and make them pass as stopwords
all_tokens = []
for doc in docs:
    tokens = my_tokenizer(doc)
    all_tokens.append(tokens)

# Features version 1: word embedding training
model = Word2Vec(size=200, window=5, min_count=2) # hacer una busqueda de parametros aqui tambien
model.build_vocab(all_tokens)
model.train(all_tokens, total_examples=model.corpus_count, epochs=model.iter)
word_vectors = model.wv

# TODO: chequear a mano algunas distancias: hombre-mujer mujer-violencia violencia-hombre

# Create the matrix of embeddings for each word
# set the matrix dimensions and creates matrix


index_word_map_embeddings = []
all_words = []
for t in all_tokens: # Now we want each token independent of their document
    for n in t:
        if n not in word_vectors.vocab:
            pass
        else:
            all_words.append(n)
all_words = set(all_words)
N = len(all_words) # N is the number of unique tokens
D = model.vector_size # D is the vector size of the embedding
X = np.zeros((N, D)) # terms will go along rows, documents along columns
i = 0
for token in all_words: # Index of the words to retrieve them latter
    a = model.wv[token]
    index_word_map_embeddings.append(token)
    X[i,:] = a
    i += 1
print ('Embedding features')
print ('Samples (unique words): '+str(N))
print ('Feature vector: '+str(D))

# We will reduce the dimensionality of the embeddings with t-SNE
# For further information read:
# To check the function parameters: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
reducer = TSNE(perplexity=40) # estimar
Z = reducer.fit_transform(X)

# Run Kmeans TODO: we need to try different Ks and plot that against the cost

y = []
for k in range(5, 500, 5):
    print ('k', k)
    _, _, costs, X = soft_k_means(Z, k, index_word_map_embeddings, show_plots=False) # it chooses a K given the size of the vocab
    y.append(costs[-1])

axes = plt.gca()
axes.set_xlim(1, len(y))
plt.plot(y)
print (y)
print (y.index(min(y)),min(y))
slopes = [x - z for x, z in zip(y[:-1], y[1:])]
p = slopes.index(max(slopes))
print ('slope')
print (slopes)

p = y[p]
i = y.index(p)
print (p, i)
plt.plot(i,p, marker='x')
plt.annotate(s='k='+str(i),xy=(i,p),xytext=(i,p))
plt.plot(y.index(min(y)),min(y), marker='o')
print (y.index(min(y)), min(y))
plt.annotate(s='k='+str(y.index(min(y))),xy=(y.index(min(y)),min(y)),xytext=(y.index(min(y)),min(y)))
plt.title("Costs")
plt.show()
plt.savefig("cost.png")
