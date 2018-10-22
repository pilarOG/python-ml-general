# -*- coding: utf-8 -*-

# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
  def __init__(self, m): # m = true mean
    self.m = m
    self.mean = 0 # estimate of the bandit mean
    self.N = 0

  def pull(self): # pulling the bandit's arm
    return np.random.randn() + self.m # bandit's reward is a gaussian with unit variance

  def update(self, x): # x = latest sample received from the bandit
    self.N += 1
    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x # update equation that efficiently takes the last mean and updates
                                                          # instead of re calculating the mean of all the samples each time


def run_experiment(m1, m2, m3, eps, N): # three means to compare/ 3 bandits; N = number of times we play
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

  data = np.empty(N) # keep results

  for i in range(N):
    # epsilon greedy
    p = np.random.random() # random number between 0 and 1
    if p < eps:
      j = np.random.choice(3)  # choose a bandit in random
    else:
      j = np.argmax([b.mean for b in bandits])  # choose bandit with the current best sample mean

    x = bandits[j].pull() # pull
    bandits[j].update(x)  # update the bandit with the reward

    # for the plot
    data[i] = x
  cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  plt.plot(np.ones(N)*m1)
  plt.plot(np.ones(N)*m2)
  plt.plot(np.ones(N)*m3)
  plt.xscale('log')
  plt.show()

  for b in bandits:             # print estiamted means
    print(b.mean)

  return cumulative_average # array of cumulative means per bandit after every play

if __name__ == '__main__':
  c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)          # we use same means but change epsilon to compare
  c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
  c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)

  # plot the cumulative averages together, the three experiments

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(c_05, label='eps = 0.05')
  plt.plot(c_01, label='eps = 0.01')
  plt.legend()
  plt.show()
