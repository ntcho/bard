import pandas as pd
import numpy as np
import sklearn.metrics

def read_data(filename):
    #reads in a csv and segments the data
    #randomizes the order of the data, then splits it into different sets
    #returns separate inputs (x) and outputs (y) for each of training, test, and validation
    #also returns a list of column names, which may be useful for determining heavily weighted features
    df = pd.read_csv(filename)
    data = df.to_numpy()
    np.random.shuffle(data)
    test_size = int(data.shape[0]/10)
    data_test = data[:test_size]
    data_val = data[test_size:2*test_size]
    data_train = data[2*test_size:]
    x_train = data_train[:,1:]
    y_train = data_train[:,0]
    x_val = data_val[:,1:]
    y_val = data_val[:,0]
    x_test = data_test[:,1:]
    y_test = data_test[:,0]
    # x: data point, y: data label
    return x_train, y_train, x_val, y_val, x_test, y_test, df.columns.values

def train_perceptron(x_train, y_train, x_val, y_val):
    weights = np.zeros(x_train.shape[1])
    bias = 0
    
    prev_weights = weights
    prev_accuracy = 0.0 # higher the better
    iteration = 1
    
    while True:
        for i in range(len(x_train)):
            if np.sign(x_train[i].dot(weights) + bias) != y_train[i]:
                weights = weights + y_train[i] * x_train[i]
                bias = bias + y_train[i]
        
        print(f"Train iteration {iteration}:")
        accuracy, precision, recall = evaluate_perceptron(x_val, y_val, weights, bias)
        
        if accuracy <= prev_accuracy:
            weights = prev_weights # revert to previous weights
            break
        
        prev_weights = weights
        prev_accuracy = accuracy
        iteration = iteration + 1
    
    return weights, bias

def evaluate_perceptron(x, y, weights, bias):
    #takes in a matrix of feature vectors x and a vector of class labels y
    #also takes a vector weights and a scalar bias for the classifier
    #returns the evaluations on the data (x, y) of the perceptron classifier
    y_pred = np.sign(x.dot(weights) + bias)
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    precision = sklearn.metrics.precision_score(y, y_pred, average="macro", zero_division=0)
    recall = sklearn.metrics.recall_score(y, y_pred, average="macro", zero_division=0)
    
    print(f"  accuracy={accuracy:.5f}, \n  precision={precision:.5f} \n  recall={recall:.5f}")
    return accuracy, precision, recall

def rank_features(weights, feats):
    #takes in a weight vector and an array of feature names
    #returns a sorted array of features, sorted from most negatively weighted to most positively weighted
    #note that feats MUST be a numpy array of the same length as weights
    #if feats[i] does not correspond to weights[i], this will not return accurate results
    imp = np.argsort(weights)
    return list(zip(feats[imp], weights[imp]))

def test_perceptron():
    x_train, y_train, x_val, y_val, x_test, y_test, feats = read_data("mushrooms_perceptron.csv")

    # train perceptron
    weights, bias = train_perceptron(x_train, y_train, x_val, y_val)
    print(f"Weights: {weights}")
    print(f"Bias: {bias:.2f}")

    # test perceptron
    print(f"Training set:")
    evaluate_perceptron(x_train, y_train, weights, bias)

    print(f"Validation set:")
    evaluate_perceptron(x_val, y_val, weights, bias)

    print(f"Test set:")
    accuracy, precision, recall = evaluate_perceptron(x_test, y_test, weights, bias)

    ranked_feats = rank_features(weights, feats)
    print(f"Top 3 indicative feats for edible mushrooms (y = -1):\n  {ranked_feats[:3]}")
    print(f"Top 3 indicative feats for poisionous mushrooms (y = 1):\n  {ranked_feats[::-1][:3]}")
    print()
    
    return weights, bias, accuracy, precision, recall


x_train, y_train, x_val, y_val, x_test, y_test, feats = read_data("mushrooms_perceptron.csv")

iteration = 1 # set it to > 1 to see average performance

if iteration == 1:
    # measure individual performance
    test_perceptron()
else:
    # measure average performance
    weights_sum = np.zeros(x_train.shape[1])
    bias_sum = 0
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0

    for i in range(iteration):
        print(f">> Test iteration {i}:")
        weights, bias, accuracy, precision, recall = test_perceptron()
        weights_sum += weights
        bias_sum += bias
        accuracy_sum += accuracy
        precision_sum += precision
        recall_sum += recall

    print(f"Average weights: {weights_sum / iteration}")
    print(f"Average bias: {bias_sum / iteration:.5f}")
    print(f"Average test accuracy: {accuracy_sum / iteration:.5f}")
    print(f"Average test precision: {precision_sum / iteration:.5f}")
    print(f"Average test recall: {recall_sum / iteration:.5f}")

    ranked_feats = rank_features(weights_sum / iteration, feats)
    print(f"Average top 3 indicative feats for edible mushrooms (y = -1):\n  {ranked_feats[:3]}")
    print(f"Average top 3 indicative feats for poisionous mushrooms (y = 1):\n  {ranked_feats[::-1][:3]}")
