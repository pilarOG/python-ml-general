# -*- coding: utf-8 -*-

# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
from __future__ import print_function, division, unicode_literals
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
import codecs
from nltk.stem import SnowballStemmer
import unicodedata

def my_tokenizer(s):
    s = s.lower().replace('-', '').replace(')', '').replace('(', '').replace('\"', '').replace(';', '').replace(':', '').replace(u'á', 'a').replace(u'é', 'e').replace(u'í', 'i').replace(u'ó', 'o').replace(u'ú', 'u').replace(u'ñ', 'n').replace('\u201d', '').replace('\u201c', '').replace('\xa0','')
    # lemmatization does not work very well for spanish, let's stemm
    tokens = []
    #tokens = [snowball_stemmer.stem(t) for t in s.split(' ')]
    tokens = [t for t in s.split(' ')]
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
    return tokens



# now let's create our input matrices - just indicator variables for this example - works better than proportions
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    return x

def d(u, v):
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

def plot_k_means(X, K, index_word_map, max_iter=50, beta=1.0, show_plots=True):
    N, D = X.shape # X has the shape of N (number of docs) by D (the vocabulary)
    print ('N D', N, D)
    M = np.zeros((K, D)) # create mean matrix, K x D
    R = np.zeros((N, K)) # create responsabilities of each sample to each K
    print ('R dimensions',R.shape)
    exponents = np.empty((N, K)) # calculation of R

    # initialize M to random from N, therefore, results might change each time
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter) # cost vector to lenght of the number of iterations, the cots at each iteration
    for i in range(max_iter):
        # step 1: determine assignments / resposibilities
        for k in range(K):
            for n in range(N):
                #print (X[n]) # vector for each N sample doc
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)
        # step 2: recalculate means
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()

        costs[i] = cost(X, R, M)
        print ('cost', costs[i])
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 10e-5: # cost does not change too much anymore, even if we have not reached the max number of iterations
                break

    if show_plots:
        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.figure(figsize=(80.0, 80.0))
        plt.scatter(X[:,0], X[:,1], s=300, alpha=0.9, c=colors)
        #annotate1(X, index_word_map)
        plt.savefig("test.png")


    # print out the clusters
    hard_responsibilities = np.argmax(R, axis=1) # is an N-size array of cluster identities
    print (len(hard_responsibilities))
    # let's "reverse" the order so it's cluster identity -> word index
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
      word = index_word_map[i]
      cluster = hard_responsibilities[i]
      if cluster not in cluster2word:
        cluster2word[cluster] = []
      cluster2word[cluster].append(word)
    l = 0
    # print out the words grouped by cluster
    for cluster, wordlist in cluster2word.items():
      print("cluster", cluster, "->", wordlist)
      l += len(wordlist)
    print (l)
    return M, R

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


# MAIN

# Load the data and split at some naive sentence boundaries
titles = [line.rstrip() for line in codecs.open('noticias-tercera.txt', encoding='utf-8').read().replace(',', '\n').replace('.', '\n').split('\n')]

# Some spanish stopwords
spanish_stopwords = [u'del', u'la', u'de', u'y', u'en', u'un', u'el', u'la', u'un', u'una', u'los',
         u'ls', u'las', u'unos', u'unas', u'uns', u'del', u'dl', u'al', u'la', u'el',
         u'las', u'los', u'y', u'con', u'de', u'para', u'por', u'al', u'a', u'ha',
         u'desde', u'una', u'un', u'o', u'en', u'me', u'y', u'se', u'que', u'como', u'porque', u'este']

stopwords = set(spanish_stopwords)

# add stopwords with least frequency

frequencies = {}
all_tokens = []

for title in titles:
    tokens = my_tokenizer(title)
    all_tokens.append(tokens)
    for t in tokens:
        if t not in frequencies:
            frequencies[t] = 1
        else:
            frequencies[t] += 1

new_tokens = []
for tokens in all_tokens:
    for t in tokens:
        if frequencies[t] < 3:
            pass
        else:
            stopwords.add(t.lower())

print (stopwords)

# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
print("num docs:", len(titles))
print("first doc:", titles[0])
for title in titles:
    all_titles.append(title)
    tokens = my_tokenizer(title)
    all_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            index_word_map.append(token)


# la input matrix es una matriz de N muestras 1040, donde las muestras son palabras
# y D es un vector de tamaño 462, número de documentos, con 1 o 0 si la palabra está o no en el documento

# set the matrix dimensions and creates matrix
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N)) # terms will go along rows, documents along columns
i = 0
for tokens in all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1
vocabulary_size = current_index
print("vocab size:", current_index)

# transform the frequency matrix to tfidf TODO: why not countvectorizer?
# why would tfidf represent
transformer = TfidfTransformer() #
X = transformer.fit_transform(X).toarray()

#TODO: visualize TSNE with tfidf and embeddings
reducer = TSNE()
Z = reducer.fit_transform(X)
# Run Kmeans TODO: we need to try different Ks and plot that against the cost
#plot_k_means(Z[:,:2], vocabulary_size//10, index_word_map, show_plots=True) # it chooses a K given the size of the vocab
M, R = plot_k_means(Z, vocabulary_size//20, index_word_map, show_plots=False) # it chooses a K given the size of the vocab
#print (M, R)
