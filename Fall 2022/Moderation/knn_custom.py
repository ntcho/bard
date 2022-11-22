import numpy as np

# returns euclidian distance between data points
def distance(a, b):
    # should be np.sqrt(np.sum(...)), but omitted for performance
    # WARNING: make sure to cast the data into correct data type as uint8 will overflow!
    return np.sum(np.square(a.astype(int) - b.astype(int)))


# test kNN with multiple k values
def kNN(k, test, train):
    y_pred = np.empty(shape=(test.shape[0]), dtype=int)

    # iterate through test set
    for i, d_test in enumerate(test):
        # create array of [[distance, label], ...] for every point in training set
        distances = np.array(
            [[distance(d_test[1:], d_train[1:]), d_train[0]] for d_train in train]
        )

        # k-th closest data points
        # sort with column 0 (distance), extract k rows
        y_pred_knn = distances[distances[:, 0].argsort()][:k]

        # find most common label from kNN's labels
        values, counts = np.unique(y_pred_knn, return_counts=True)
        y_pred[i] = values[np.argmax(counts)]

    return y_pred
