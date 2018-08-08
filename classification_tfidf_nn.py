import pandas as pd
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import L1L2
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import TruncatedSVD
import keras
import numpy as np
from numpy import argmax
import keras_metrics
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf

FEATURES = 10000
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
        # temp = [lemma.stem(word) for word in temp.split()]
        cleaned_phrase.append(' '.join(temp))
    return cleaned_phrase

def vector_dimension_reduction(X):
    ## PCA/LDA both use SVD - PCA find the best variance independent of class while LDA is to find vectors with best variance based on class
    ## We are going to use LSA/SVD
    svd = TruncatedSVD(n_components=COMP, n_iter=7, random_state=seed)
    feature_vector = svd.fit_transform(X)
    # with open("svd_model.pickle", "wb") as fp:
    #     cPickle.dump(svd, fp, protocol=2)
    return feature_vector

def batch_generator_shuffle(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:]
        y_batch = y_data[index_batch,:]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            np.random.shuffle(index)
            counter=0

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def auc_pr(y_true, y_pred, curve='PR'):
    return tf.metrics.auc(y_true, y_pred, curve=curve)

if __name__ == "__main__":
    df = pd.read_csv('dataset/train.csv', encoding = 'unicode_escape')
    df = df[['TRANS_CONV_TEXT', 'Patient_Tag']]

    df_test = pd.read_csv('dataset/test.csv', encoding = 'unicode_escape')
    df_test = df_test[['Source', 'Host', 'Link', 'Date(ET)', 'Time(ET)', 'time(GMT)', 'Title', 'TRANS_CONV_TEXT']]

    ## Checking data size for category 0 and 1
    # print(df['Patient_Tag'].value_counts())

    df.dropna(how='any', inplace=True)
    # df_test.dropna(how='any', inplace=True)

    ## Checking data size for category 0 and 1
    # print(df['Patient_Tag'].value_counts())

    ## extracting training data
    X = df['TRANS_CONV_TEXT']
    Y = df['Patient_Tag']

    X_test = df_test['TRANS_CONV_TEXT']

    Y = to_categorical(Y, num_classes=2)
    print (Y.shape)
    # Y.dump("y.txt")

    # print (X_train[5])
    # print (Y_train[5])

    ## Stemmer to convert words into there root from not required in this case since lemma can be used
    stemmer = SnowballStemmer("english")

    ## Lemmatizer to convert wrods into base form like words to word
    lemma = WordNetLemmatizer()

    ## Remove
    X = clean_data(X, lemma)
    X_test = clean_data(X_test, lemma)

    ## BOW Method - Dictionary is build from the data based on the x number of important words/features

    ## Applying TFIDF and keep top 1000 important words and throwing all the other words. Each word represent a dimension.
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=FEATURES, stop_words='english', ngram_range=(1, 3))
    X_vec = tfidf.fit_transform(X).toarray()
    X_test = tfidf.transform(X_test).toarray()
    # print (X_vec.shape)
    # for i in range(X_vec.shape[0]):
    #     print (X_vec[i])

    ## Applying LSI to extract most important features Since we are using NN we need more features for the neurons to learn so not required
    # X_vec = vector_dimension_reduction(X_vec)
    # X_test = vector_dimension_reduction(X_test)

    ## Cross Validation Set creation by train validation split
    X_train, X_val, y_train, y_val = train_test_split(X_vec, Y, test_size=0.33, random_state=seed)
    print (X_train.shape)
    print (X_test.shape)
    # print(y_train.shape)

    ## Using shallow NN for binary classification

    ## Custom Adam so that we can play around with the learning rate
    custom_adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=FEATURES))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, # output dim is 2, one score per each class
                    activation='softmax',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.4)))

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model.compile(optimizer=custom_adam,
                  loss='categorical_crossentropy',
                  metrics=[auc_pr, 'accuracy'])

    # model.fit(X_vec, Y, epochs=50, validation_split=0.2)

    ## Adding shuffling for better generalization and higher accuracy
    model.fit_generator(generator=batch_generator_shuffle(X_train, y_train, 32),
                          epochs=7, validation_data=(X_val, y_val),
                          steps_per_epoch=X_train.shape[0] / 32)

    scores = model.evaluate(X_vec, Y)
    print ('\n')
    print (scores)

    output = model.predict(X_test, 32)
    # print (output[:10])
    df_test['Patient_Tag'] = np.argmax(output, axis=1)
    df_test['Index'] = list(range(1, len(output)+1))
    df_test_out = df_test[['Index','Patient_Tag']].set_index('Index')
    # df_test_out = df_test[['Patient_Tag']]
    df_test_out.to_csv('aakash.agarwal2602@gmail.com.csv', columns=['Patient_Tag'] )
