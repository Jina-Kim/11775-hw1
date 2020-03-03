#!/bin/python
import numpy as np
import os
import cPickle

from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import sys
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk import word_tokenize


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: {0} feat_dim".format(sys.argv[0])
        print "feat_dim -- dim of features; provided just for debugging"
        exit(1)

    features = int(sys.argv[1])

    files = os.listdir("../asrs")

    df = pd.read_csv("asrs.all.preprocess.csv")
    df = df.fillna("")

    train_df = pd.read_csv("../all_trn.lst", names=['id', 'label'], sep=" ")
    train_df = train_df.merge(df, left_on='id', right_on='video_id', how='left')
    train_df = train_df.drop('video_id', axis=1)
    train_df['text_after'] = train_df['text_after'].fillna("")

    if features == 1:
        lines = df['text_after'].values.tolist()
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(lines)]

        model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, workers=2)
        model.build_vocab(tagged_data)

        for epoch in range(100):
            model.train(tagged_data, total_examples=model.corpus_count, epochs=4)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        li = []
        for i in range(len(lines)):
            x = model.docvecs[i]
            li.append(x/np.linalg.norm(x))
        vect = np.asarray(li)

    else:
        clf = TfidfVectorizer(max_features=features).fit(train_df['text_after'])
        # clf = CountVectorizer(max_features=features)
        vect = clf.transform(df['text_after']).toarray()

    new_df = pd.DataFrame(data=vect, index=[i for i in range(vect.shape[0])], columns=['m'+str(i) for i in range(vect.shape[1])])
    new_df["video_id"] = df['video_id']

    new_df.to_csv("asrfeat/asrfeat."+str(features)+".csv", index=False)

    print "ASR features generated successfully!"
