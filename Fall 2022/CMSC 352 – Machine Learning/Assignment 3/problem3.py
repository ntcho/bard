import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

from timeit import default_timer as timer

timestamp = timer()


def check_elapsed(msg):
    global timestamp
    print(f"{msg:<70} [{(timer() - timestamp):.3f}s elapsed]")
    timestamp = timer()


def evaluate(model_name, y_true, y_pred):
    print(f"[{model_name}] Performance analysis:")
    print(f"accuracy:  {metrics.accuracy_score(y_true, y_pred):.3f}")
    print(f"precision: {metrics.precision_score(y_true, y_pred):.3f}")
    print(f"recall:    {metrics.recall_score(y_true, y_pred):.3f}")


def get_word_filter(
    filename_vocab="multi_vocab",
    filename_positive="positive-words.txt",
    filename_negative="negative-words.txt",
):
    vocabs = open(filename_vocab, "r").readlines()
    positives = open(filename_positive, "r").readlines()
    negatives = open(filename_negative, "r").readlines()

    sentiment_indexes = []
    positive_indexes = []
    negative_indexes = []

    for i, w in enumerate(vocabs):
        if w in positives:
            positive_indexes.append(i)
            sentiment_indexes.append(i)
            continue
        if w in negatives:
            negative_indexes.append(i)
            sentiment_indexes.append(i)

    # index of positive and negative features
    return sentiment_indexes, positive_indexes, negative_indexes


def get_sentiment_word_count(x_feats, positive_indexes, negative_indexes):
    word_counts = np.sum(x_feats, axis=1)
    positive_counts = np.sum(x_feats[:, positive_indexes], axis=1)
    negative_counts = np.sum(x_feats[:, negative_indexes], axis=1)
    netural_word_count = word_counts - (positive_counts + negative_counts)

    return word_counts, positive_counts, negative_counts, netural_word_count


### Read data
df = pd.read_pickle("IMDB_BOW.pkl")  # 50,000 rows, 6,666 columns

check_elapsed("Completed reading IMDB_BOW.pkl")

data = df.to_numpy()

### Filter features which only has sentiment words
sentiment_features, positive_indexes, negative_indexes = get_word_filter()
(
    word_counts,
    positive_counts,
    negative_counts,
    netural_word_count,
) = get_sentiment_word_count(data[:, 1:], positive_indexes, negative_indexes)

data = np.column_stack(  # add columns
    (
        data[:, 0],  # labels
        data[:, 1:],  # all features
        # data[:, 30:1000],  # only leave top 1000 words
        # data[:, 1:][:, sentiment_features],  # only leave sentiment vocabulary features
        # word_counts,  # number of words included in BOW in review
        # positive_counts,  # number of positive words
        # negative_counts,  # number of negative words
        # netural_word_count,  # number of netural words
    )
)

print(f"size={np.shape(data)}")

np.random.shuffle(data)
check_elapsed("Completed shuffling dataset")

### Split data
scaler = StandardScaler()
x = scaler.fit_transform(data[:, 1:])
test_size = int(data.shape[0] / 10)
check_elapsed("Completed normalizing features")

# split into 80% training, 10% validation and 10% test data
x_train = x[2 * test_size :]
x_val = x[test_size : 2 * test_size]
x_test = x[:test_size]

y_train = data[2 * test_size :, 0]
y_val = data[test_size : 2 * test_size, 0]
y_test = data[:test_size, 0]

check_elapsed("Completed splitting training, validation and test set")

# set to true to show verbose outputs for classifier
verbose = True


"""
Support Vector Classifier
"""
print(f"Started training support vector")
clf_svc = SVC(verbose=verbose, cache_size=8192)  # using default gamma
clf_svc.fit(x_train, y_train)

print()
check_elapsed(f"Completed training support vector")
evaluate("support vector (training set)", y_train, clf_svc.predict(x_train))
evaluate("support vector (validation set)", y_val, clf_svc.predict(x_val))
evaluate("support vector (test set)", y_test, clf_svc.predict(y_test))
check_elapsed("Completed validating support vector")
