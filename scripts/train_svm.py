#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.linear_model import LogisticRegression
import cPickle
import sys
import pandas as pd
import xgboost as xgb


# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    # python scripts/train_svm.py $event "kmeans/" $feat_dim_mfcc mfcc_pred/svm.$event.model
    df = pd.read_csv(feat_dir+feat_dir[:-1]+"."+str(feat_dim)+".csv") # video_id, mfcc
    train_df = pd.read_csv("../all_trn.lst", names=['id', 'label'], sep=" ")

    train_df = train_df.merge(df, left_on='id', right_on='video_id', how='left')
    train_df = train_df.drop('video_id', axis=1)
    train_df = pd.get_dummies(train_df, columns=['label'])
    train_df.iloc[:,1:-3] = train_df.iloc[:,1:-3].fillna(0)

    X = train_df.iloc[:,1:-3]
    y = train_df['label_'+event_name]

    # clf = SVC(probability=True, gamma='auto')
    clf = xgb.XGBClassifier()
    clf = xgb.XGBClassifier(learning_rate=0.01,
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      n_estimators=1000,
                      reg_alpha = 0.3,
                      max_depth=4)
    clf.fit(X, y)


    with open(output_file, 'wb') as f:
        cPickle.dump(clf, f)


    print 'SVM trained successfully for event %s!' % (event_name)
