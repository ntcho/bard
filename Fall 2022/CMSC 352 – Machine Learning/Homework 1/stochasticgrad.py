import pandas as pd
import numpy as np

df = pd.read_csv("geyser.csv")
all_data = df.to_numpy()


def compute_cost(data, weights):
    pred = weights[0] + weights[1] * data[:, 0]
    cost = 0.5 * np.sum(np.square(pred - data[:, 1]))
    return cost


def compute_grad(data, weights):
    # Using the formula we discussed in class for computing the derivative, fill in this function
    # The argument data is a two column numpy array, with one data point per row
    # The first column contains the duration values, the second the interval values
    # The argument weights is the current weights
    # Return a numpy array of length 2 with the partial derivatives of the cost function

    deriv = None

    for x, y in data:
        hx = weights[0] + weights[1] * x
        deriv = np.array([hx - y, x * (hx - y)])
        weights -= lr * deriv

    return deriv  # return last derivative


def print_info(i, weights, cost):
    print("Iteration %i" % i)
    print("Weights are currently <%f, %f>" % (weights[0], weights[1]))
    print("Current cost: %f" % cost)


lr = 0.00001  # learning rate--keep this small
stop = False  # becomes true when the cost stops decreasing
i = 1  # count how many iterations in we are
w = np.array([0.0, 0.0])  # initialize to zero
cost_decrease = np.inf
cost = np.inf

while cost_decrease > 0.001:
    deriv = compute_grad(all_data, w)
    w -= lr * deriv
    newcost = compute_cost(all_data, w)
    cost_decrease = cost - newcost

    if cost_decrease <= 0:
        break

    cost = newcost
    print_info(i, w, cost)
    i += 1
