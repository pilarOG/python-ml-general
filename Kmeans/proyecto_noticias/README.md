# Clustering words in the news: an example with femicide news

I have been taking the Unsupervised ML course in Udemy which you can find here:<br>
https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python <br>
And here you can find the original code:<br>
https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python

At the end of the k-means section an example code is taught where soft k-means is used to find clusters of words in documents. This is my version to understand and apply that code.

<b>Data</b>: In this version the data is taken from 10 news from "La Tercera", a chilean newspaper, that are retrieve when you search "femicide" (in Spanish, of course). I wanted to try the code on relevant data to see what kind of words where clusters together in this relevant topic. <br><br>
<b>Features</b>: Each sentence in the news data is embedded using Gensim Word2Vec. Then each word is represented with the embedding vector.<br><br>
<b>Visualization</b>: The resulting embeddings are visualized using t-SNE, reducing dimensionality to 2.<br><br>
<b>Soft k-means</b>: The reduced vectors are used as input to soft K-means to find clusters of the words. The goal is to find which words tend to appear together or related and with such representation have some insights about the topic of the data and the editor's view of the topic. In the code you can find further details about the algorithm.<br><br>
<b>Finding K</b>: in the folder you can find the plot "best-K.png" which shows how I tried different K's until finding the optimal one which is the one used in the example code.

### Requirements

To use it you need to run:
```
pip install -r requirements.txt
```

### Experiments

All hyperparameters will be tuned given the cost achieved by K. As this is an unsupervised learning task we don't have a gold standard to tune our parameters against. Therefore, the parameter that get's the lower k-means cost will be used. This is not a proper grid-seach as the hyperparameters were tuned one at the time and in the following order (the hyperparemeters previous to k used a default k = 50). Problem of random initilization.

Hyperparameter tuning of embeddings:<br><br>

Word2Vec is an unsupervised representation of words in a context (distributed word representation).
For the original source you can read: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

<b>Vector size</b>: it determines the dimensionality of the word vector. <br><br> (window=3, min_count=2)

150 = cost 5977.69716138
200 = cost 4621.96378964
250 = cost 5205.64797857
300 = cost 5792.35941545

<b>Window</b><br><br> (size=200, min_count=2)

1 = cost 5073.87229277
2 = cost 4800.43862261
3 = cost 5449.21504495
4 = cost 4691.96661704
5 = cost 4485.86749579
6 = cost 4639.6917254

Hyperparameters tuning of t-SNE:

<b>Perplexity</b> A mayor perplexidad se van juntando más los clusters <br><br>

20 = cost 5013.82773706
25 = cost 3869.31913356
30 = cost 3305.60081724
35 = cost 2531.70231462
40 = cost 2206.22883093
45 = cost 1522.89252018
50 = cost 1077.59035971
55 = cost 633.523643424
60 = cost 425.583739954

Hyperparameter tuning of K:<br><br>

<b>K</b>

### Results

I give here a brief analysis on the results.
