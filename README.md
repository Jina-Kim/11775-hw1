## HW1: Audio-based Multimedia Event Detection
### CMU 11-775 Large-Scale Multimedia Analysis (Spring2020)

1. Pipeline
* Feature extraction (run.feature.sh)
  * MFCC
    - extract audio (.wav) from video files and MFCC features
    - train k-means model and reduce MFCC feature dimension into k-dim vector (count the predicted clusters in each video files and normalize the vector)
  * ASR Transcriptions
    - preprocess transcriptions txt files and convert the dialog in each video into the word vectors (CountVectorizer, TfidfVectorizer, and Doc2Vec)
  * MFCC + ASR Transcriptions
    - concatenate two extracted features

* Multimedia Event Detection Classification (run.med.sh)
  * Classifiers
    - train each classifier (SVM and XGBoost) in train set and calculate average precision on validation/test set
