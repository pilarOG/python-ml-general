from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import sys
# K means to cluster sentences from a csv

def preprocessing(data):
    # check for correct headers
    if 'sample' not in data:
        raise Exception('Data headers are wrong')
    # check for empty values
    if data.isnull().values.any():
        raise Exception('There are missing values')
    count_vectorizer = CountVectorizer() # here try any transform you want to do clustering
    return count_vectorizer.fit_transform(data['sample'])

def find_clusters(feats):
    num_clusters = 2 # change this for the number of clusters to find
    km = KMeans(n_clusters=num_clusters)
    km.fit(feats)
    clusters = km.labels_.tolist()
    return clusters

# TODO: if you want to iterate over a range of possible ks
def iterate_k(feats):
    pass


#output = pd.DataFrame(data = {'cluster':clusters,'intent':train_data['intent'],'data':train_data['input']})
#output.to_csv('chitchat_clusters_topics.csv', sep=",", encoding = 'utf-8')

if __name__=='__main__':
    if len(sys.argv) == 1: raise Exception('Run this script in your terminal with the name of your file-data as first argument')
    data = pd.read_csv(sys.argv[1], header = 0, delimiter = "\t", encoding = 'utf-8')
    feats = preprocessing(data)
    clusters = find_clusters(feats)
    # TODO: get results as a matrix
    for n in range(len(data)):
        print u'Sample \'{}\' corresponds to cluster {}'.format(data['sample'][n], clusters[n])
