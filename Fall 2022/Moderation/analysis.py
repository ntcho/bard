import numpy as np
import pandas as pd
from timeit import default_timer as timer

import knn_custom
import knn_sklearn
import knn_cv
import knn_evaluate

timestamp = timer()


def check_elapsed(msg):
    global timestamp
    print(f"{msg} [{(timer() - timestamp):.3f}s elapsed]")
    timestamp = timer()


"""
Part 1. Import data
"""
print("Part 1. Import data")

# file names to read from
input_test = "mnist_test.small.csv"  # 10MB
input_train = "mnist_train.small.csv"  # 100MB

# test set; the performance will be evaluated with this set
# `dtype=np.uint8` for memory optimization, `engine="pyarrow"` for I/O performance
test = pd.read_csv(
    input_test, delimiter=",", header=None, dtype=np.uint8, engine="pyarrow"
).values
check_elapsed(f"Completed reading test dataset. (n={len(test)})")

# train set; test set will iterate over this and compare
train = pd.read_csv(
    input_train, delimiter=",", header=None, dtype=np.uint8, engine="pyarrow"
).values
check_elapsed(f"Completed reading train dataset. (n={len(train)})")

print()
"""
Part 2. Train & test model
"""
print("Part 2. Train & test model")

# hyperparameter k, finding k nearest neighbors
k = 5

# 1. custom
y_pred_custom = knn_custom.kNN(k, test, train)
check_elapsed("[custom]  Completed training and testing kNN model.")

# 2. sklearn
y_pred_sklearn = knn_sklearn.kNN(k, test, train)
check_elapsed("[sklearn] Completed training and testing kNN model.")

# 3. cv
y_pred_cv = knn_cv.kNN(k, test, train)
check_elapsed("[cv]      Completed training and testing kNN model.")


print()
"""
Part 3. Evaluate model
"""
print("Part 3. Evaluate model")

y_true = test[:, 0]

# 1. custom
knn_evaluate.evaluate("custom", y_true, y_pred_custom)

# 2. sklearn
knn_evaluate.evaluate("sklearn", y_true, y_pred_sklearn)

# 3. cv
knn_evaluate.evaluate("cv", y_true, y_pred_cv)
