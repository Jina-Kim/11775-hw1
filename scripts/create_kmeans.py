#!/bin/python
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

import numpy as np
import pandas as pd
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))

    mfcc_files = os.listdir("mfcc")
    name = []
    vect = []

    for i, file in enumerate(mfcc_files):
        mfcc_file = "mfcc/"+file
        mfcc = np.genfromtxt(mfcc_file, delimiter=";")
        print str(i)+" "+mfcc_file+" number of audio words "+str(len(mfcc))

        mfcc_clusters = kmeans.predict(mfcc)
        freq_cluster = np.bincount(mfcc_clusters, minlength=cluster_num)

        name.append(file.split(".")[0])
        vect.append(freq_cluster / np.linalg.norm(freq_cluster))

    vect = np.asarray(vect)
    df = pd.DataFrame(data=vect, index=[i for i in range(vect.shape[0])], columns=['m'+str(i) for i in range(vect.shape[1])])
    df["video_id"] = name

    df.to_csv("kmeans/kmeans."+str(cluster_num)+".csv", index=False)

    print "K-means features generated successfully!"
