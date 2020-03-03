#!/bin/python
import os
import pandas as pd

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower()
    # text = text.translate(string.maketrans('', '', string.punctuation))
    text = text.strip()


    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    words = [PorterStemmer().stem(w) for w in words]
    words = [w for w in words if not w in stopwords.words('english')]

    text = " ".join(words)

    return text

if __name__ == '__main__':

    files = os.listdir("../asrs")
    files = [x for x in files if "txt" in x]

    id = []
    txt = []
    for txt_file in files:
        with open("../asrs/"+txt_file,"r") as f:
            lines = f.read()

        id.append(txt_file.split(".")[0])
        txt.append(lines.strip())

    df = pd.DataFrame({"video_id":id,"text":txt,})
    df['text_after'] = df['text'].map(lambda x: preprocess_text(x))
    df.to_csv("asrs.all.preprocess.csv", index=False)
    print "ASR features preprocessed successfully!"
