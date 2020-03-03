#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import pandas as pd

if __name__ == '__main__':

    kmean_df = pd.read_csv("kmeans/kmeans.100.csv") # 2857
    asr_df = pd.read_csv("asrfeat/asrfeat.1.csv") # 2226
    # print asr_df.shape, kmean_df.shape

    new_df = kmean_df.merge(asr_df, left_on='video_id', right_on='video_id', how='left')
    new_df = new_df.drop('video_id', axis=1)
    new_df['video_id'] = kmean_df['video_id']
    new_df.iloc[:,:-1] = new_df.iloc[:,:-1].fillna(0)

    new_df.to_csv("mixed/mixed.0.csv", index=False)

    print new_df.head()

    print "Mixed features generated successfully!"
