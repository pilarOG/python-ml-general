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
  def __init__(self, m):
    self.m = m # original "true" mean
    self.mean = 0 # estimate of the bandit's mean
    self.N = 0 # number of samples/pulls taken so far

  def pull(self): # pulling the bandit's arm
    return np.random.randn() + self.m # bandit's reward is sampled from a gaussian (with mean 0 and variance 1) and sum to the mean "original mean" specified at the beggining
                                      # as we are summing this "true mean" over time the estimated mean will be closer to it at each time we pull that one bandit

  def update(self, x): # x = latest sample received from the bandit's pull function
    self.N += 1
    self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x # update equation that efficiently takes the last mean and updates
                                                          # instead of re-calculating the mean of all the samples each time

def run_experiment(m1, m2, m3, eps, N): # three "true" means to compare / 3 bandits; N = number of times we play
  bandits = [Bandit(m1), Bandit(m2), Bandit(m3)] # Set up thre three bandits
  data = np.empty(N)

  for i in range(N):

    print ('Bandit 1 estimated mean: '+str(bandits[0].mean))
    print ('Bandit 2 pulled '+str(bandits[0].N)+' times')
    print ('Bandit 2 estimated mean: '+str(bandits[1].mean))
    print ('Bandit 2 pulled '+str(bandits[1].N)+' times')
    print ('Bandit 3 estimated mean: '+str(bandits[2].mean))
    print ('Bandit 3 pulled '+str(bandits[2].N)+' times')

    # epsilon greedy
    p = np.random.random() # random number between 0 and 1
    if p < eps: # if p smaller than epsilon, go to explore
      j = np.random.choice(3)  # choose a bandit in random
      print ('Explore Bandit '+str(j+1))
    else: # else, use Bandit with best mean so far
      j = np.argmax([b.mean for b in bandits])  # choose bandit with the current best estimated mean
      print ('Best Bandit so far chosen: Bandit '+str(j+1))

    x = bandits[j].pull() # pull
    print ('Bandit '+str(j+1)+' reward is: '+str(x))
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

# epsilon is the probability of exploration, the larger the epsilon the more we will go to explore
if __name__ == '__main__':
  c_1 = run_experiment(1.0, 5.0, 8.0, 0.1, 1000)          # we use same means but change epsilon to compare
  c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 1000)
  c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 1000)

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
