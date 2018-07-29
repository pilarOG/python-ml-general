# -*- coding: utf-8 -*-

# This code belong to Unsupervised Learning course in Udemy
# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

############################################################################
# Notes:
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


import numpy as np
import matplotlib.pyplot as plt


def d(u, v):
    diff = u - v
    return diff.dot(diff) # dot product of the values gives the square distance


def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        # method 1
        # for n in range(len(X)):
        #     cost += R[n,k]*d(M[k], X[n]) # square distance between samples and means weighted by the responsabilities

        # method 2
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost


def plot_k_means(X, K, max_iter=20, beta=1.0, show_plots=True):
    N, D = X.shape
    M = np.zeros((K, D))
    # R = np.zeros((N, K))
    exponents = np.empty((N, K))

    # initialize M to random, e.g. the centers are at first selected randomly from the input data (X)
    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = np.zeros(max_iter) # cost progress at each iteration



    # Main algorithm
    for i in range(max_iter):
        # STEP 1: determine assignments / resposibilities or E-step > centers to soft clusters: After centers have been selected,
        # assign each data point a “responsibility” value for each cluster, where higher values correspond to stronger cluster membership.
        for k in range(K):
            for n in range(N):
                # R[n,k] = np.exp(-beta*d(M[k], X[n])) / np.sum( np.exp(-beta*d(M[j], X[n])) for j in range(K) )
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))

        R = exponents / exponents.sum(axis=1, keepdims=True)
        # assert(np.abs(R - R2).sum() < 1e-10)

        # STEP 2: recalculate means or M-step > soft clusters to centers: After data points have been assigned to soft clusters, compute new centers.
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()
        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 1e-5: # if the cost have not change much between iterations, stop
                break


    # Plots
    if show_plots:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()

    return M, R


def get_simple_data():
    # assume 3 means
    D = 2 # dimensionality of the data
    s = 4 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0]) # three means or centre of the clusters, or 3 = K
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900 # number of samples
    X = np.zeros((N, D)) # input data
    X[:300, :] = np.random.randn(300, D) + mu1 # 300 samples for each mean, adds random noise to the means
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3
    return X


def main():
    X = get_simple_data()

    # what does it look like without clustering?
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    K = 3 # luckily, we already know this
    plot_k_means(X, K)

    K = 5 # what happens if we choose a "bad" K?
    plot_k_means(X, K, max_iter=30)

    K = 5 # what happens if we change beta?
    plot_k_means(X, K, max_iter=30, beta=0.3) # when beta closer to infinite, closer to "hard" k means


if __name__ == '__main__':
    main()
