# Clustering words in the news: an example with femicide news

I have been taking the Unsupervised ML course in Udemy which you can find here:<br>
https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
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

### Results

I give here a brief analysis on the results.
