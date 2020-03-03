#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import cPickle
import sys
import pandas as pd

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} event_name model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    event_name = sys.argv[1]
    model_file = sys.argv[2]
    feat_dir = sys.argv[3]
    feat_dim = int(sys.argv[4])
    output_file = sys.argv[5]

    clf = cPickle.load(open(model_file,"rb"))

    df = pd.read_csv(feat_dir+feat_dir[:-1]+"."+str(feat_dim)+".csv") # video_id, mfcc
    val_df = pd.read_csv("../all_val.lst", names=['id', 'label'], sep=" ")

    val_df = val_df.merge(df, left_on='id', right_on='video_id', how='left')
    val_df = val_df.drop('video_id', axis=1)
    val_df = pd.get_dummies(val_df, columns=['label'])

    val_df.iloc[:,1:-3] = val_df.iloc[:,1:-3].fillna(0)

    # print val_df.shape, val_df

    X = val_df.iloc[:,1:-3]
    y = val_df['label_'+event_name]

    pre_y = clf.predict_proba(X)
    # score = clf.decision_function(X)

    y_pred = np.argmax(pre_y, axis=1)
    # print "sklearn ap: ", average_precision_score(y, y_pred)
    # print
    np.savetxt(output_file, pre_y[:,1], delimiter='\n')
