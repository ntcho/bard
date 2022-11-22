import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

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


### Read data
df = pd.read_pickle("IMDB_BOW.pkl")  # 50,000 rows, 6,666 columns

check_elapsed("Completed reading IMDB_BOW.pkl")

### Split data
data = df.to_numpy()
np.random.shuffle(data)
check_elapsed("Completed shuffling dataset")

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
verbose = False


"""
K-Nearest Neighbors Classifier
"""

print("Started training k-nearest neighbors")
clf_knn = KNeighborsClassifier(n_jobs=-1)  # enable parallel processing
clf_knn.fit(x_train, y_train)

check_elapsed("Completed training k-nearest neighbors")
evaluate("k-nearest neighbors", y_val, clf_knn.predict(x_val))
check_elapsed("Completed validating k-nearest neighbors")


"""
Decision Tree Classifier
"""

print("Started training decision tree")
clf_decision_tree = DecisionTreeClassifier(max_depth=100)
clf_decision_tree.fit(x_train, y_train)

check_elapsed("Completed training decision tree")
evaluate("decision tree", y_val, clf_decision_tree.predict(x_val))
check_elapsed("Completed validating decision tree")


"""
Logistic Regression Classifier
"""

print("Started training logistic regression")
clf_logistic_regression = LogisticRegression(verbose=verbose, max_iter=500)
clf_logistic_regression.fit(x_train, y_train)

check_elapsed("Completed training logistic regression")
evaluate("logistic regression", y_val, clf_logistic_regression.predict(x_val))
check_elapsed("Completed validating logistic regression")


"""
Linear Support Vector Classifier
"""

print("Started training linear support vector")
clf_linear_svc = LinearSVC(verbose=verbose)
clf_linear_svc.fit(x_train, y_train)

print()
check_elapsed("Completed training linear support vector")
evaluate("linear support vector", y_val, clf_linear_svc.predict(x_val))
check_elapsed("Completed validating linear support vector")


"""
Support Vector Classifier
"""

print("Started training support vector")
clf_svc = SVC(verbose=verbose, cache_size=8192)

### Training with 10,000 rows
# NOTE: resized to 10,000 rows (1/4 of actual training set) due to computational complexity
# Due to complexity of this model being O(feature_size * sample_size^2) at best case,
# and O(feature_size * sample_size^3) at worst case, running this model with
# actual training set will be 16 (4^2) ~ 64 (4^3) times slower than the result shown below.
# clf_svc.fit(x_train[:10000], y_train[:10000])

### Training with 40,000 rows
clf_svc.fit(x_train, y_train)

print()
check_elapsed("Completed training support vector")
evaluate("support vector", y_val, clf_svc.predict(x_val))
check_elapsed("Completed validating support vector")


"""
Random Forest Classifier
"""

print("Started training random forest")
clf_random_forest = RandomForestClassifier(max_depth=100, verbose=verbose)
clf_random_forest.fit(x_train, y_train)

check_elapsed("Completed training random forest")
evaluate("random forest", y_val, clf_random_forest.predict(x_val))
check_elapsed("Completed validating random forest")
