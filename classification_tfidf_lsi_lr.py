import pandas as pd
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_auc_score

FEATURES = 1000
seed = 13
COMP = 300

def clean_data(phrases, lemma):
    cleaned_phrase = []

    ## for the below punctuation we need to replace it with white space
    map1 = str.maketrans('/(){}', ' ' * 5)

    ## below punctuation to be removed from data
    map2 = str.maketrans('', '', string.punctuation)
    for phrase in phrases:
        temp = phrase.lower().translate(map1)
        temp = temp.translate(map2)
        temp = [lemma.lemmatize(word) for word in temp.split()]
        cleaned_phrase.append(' '.join(temp))
        # cleaned_phrase.append(temp)
    return cleaned_phrase

def vector_dimension_reduction(X):
    ## PCA/LDA both use SVD - PCA find the best variance independent of class while LDA is to find vectors with best variance based on class
    ## We are going to use LSA/SVD
    svd = TruncatedSVD(n_components=COMP, n_iter=7, random_state=seed)
    feature_vector = svd.fit_transform(X)
    # with open("svd_model.pickle", "wb") as fp:
    #     cPickle.dump(svd, fp, protocol=2)
    return feature_vector

if __name__ == "__main__":
    df = pd.read_csv('dataset/train.csv', encoding = 'unicode_escape')
    df = df[['TRANS_CONV_TEXT', 'Patient_Tag']]

    df_test = pd.read_csv('dataset/test.csv', encoding='unicode_escape')
    df_test = df_test[['Source', 'Host', 'Link', 'Date(ET)', 'Time(ET)', 'time(GMT)', 'Title', 'TRANS_CONV_TEXT']]

    ## Checking data size for category 0 and 1
    # print(df['Patient_Tag'].value_counts())

    df.dropna(how='any', inplace=True)
    # df_test.dropna(how='all', inplace=True)

    ## Checking data size for category 0 and 1
    # print(df['Patient_Tag'].value_counts())

    ## extracting training data
    X = df['TRANS_CONV_TEXT']
    Y = df['Patient_Tag']

    # print (X_train[5])
    # print (Y_train[5])

    X_test = df_test['TRANS_CONV_TEXT']

    ## Stemmer to convert words into there root from not required in this case since lemma can be used
    stemmer = SnowballStemmer("english")

    ## Lemmatizer to convert wrods into base form like words to word
    lemma = WordNetLemmatizer()

    X = clean_data(X, lemma)
    X_test = clean_data(X_test, lemma)

    ## BOW Method - Dictionary is build from the data based on the x number of important words/features

    ## Applying TFIDF and keep top 1000 important words and throwing all the other words. Each word represent a dimension.
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=FEATURES, stop_words='english', ngram_range=(1, 3))
    X_vec = tfidf.fit_transform(X)
    X_test = tfidf.transform(X_test)
    # print (X_train.shape)

    ## Applying LSI to extract 300 most important features
    X_vec = vector_dimension_reduction(X_vec)
    X_test = vector_dimension_reduction(X_test)

    ## Cross Validation Set creation by train test split
    X_train, X_val, y_train, y_val = train_test_split(X_vec, Y, test_size=0.2, random_state=seed)

    ## Applying Logistic Regression for binary classification
    lr = linear_model.LogisticRegression(C=10, max_iter=10)
    lr.fit(X_train, y_train)
    target_names = ['class 0', 'class 1']
    print(lr.score(X_train, y_train))
    print (lr.score(X_val, y_val))

    # print(classification_report(y_train, lr.predict(X_train), target_names=target_names))
    # print(classification_report(y_val, lr.predict(X_val), target_names=target_names))

    print(roc_auc_score(y_train, lr.predict(X_train)))
    print (roc_auc_score(y_val, lr.predict(X_val)))

    output = lr.predict(X_test)
    # print(output[:10])
    df_test['Patient_Tag'] = output
    df_test['Index'] = list(range(1, len(output) + 1))
    df_test_out = df_test[['Index', 'Patient_Tag']].set_index('Index')
    df_test_out.to_csv('aakash.agarwal2602@gmail.com1.csv')

