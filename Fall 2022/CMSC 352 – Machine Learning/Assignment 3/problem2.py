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


### Read data
df = pd.read_pickle("IMDB_BOW.pkl")  # 50,000 rows, 6,666 columns

check_elapsed("Completed reading IMDB_BOW.pkl")

### Split data
data = df.to_numpy()[:5000]  # only use first 5,000 rows
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

# default gamma value with the default parameter SVC(gamma='scale')
default_gamma = 1 / (x_train.shape[1] * np.var(x_train))  # 1 / (n_features * X.var())

gamma_range = [
    default_gamma / 10,
    default_gamma / 5,
    default_gamma / 2,
    default_gamma,
    default_gamma * 2,
    default_gamma * 5,
    default_gamma * 10,
]


"""
Test 1: Test each gamma values
"""
print("Testing gamma of", gamma_range)

# test multiple gammas with SVC
for gamma in gamma_range:
    print(f"Started training support vector, gamma={gamma:.8f}")
    clf_svc = SVC(verbose=verbose, cache_size=8192, gamma=gamma)
    clf_svc.fit(x_train, y_train)

    print()
    check_elapsed(f"Completed training support vector, gamma={gamma:.8f}")
    evaluate("support vector (training set)", y_train, clf_svc.predict(x_train))
    evaluate("support vector (validation set)", y_val, clf_svc.predict(x_val))
    check_elapsed("Completed validating support vector")


"""
Test 2: Grid search all gamma values
"""
grid = GridSearchCV(
    SVC(verbose=verbose, cache_size=8192),
    param_grid=dict(gamma=gamma_range),  # range of gamma to search
    cv=StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42
    ),  # cross validation
    verbose=verbose,
    n_jobs=-1,
)

grid.fit(x_train, y_train)

print("best parameter:", grid.best_params_)
print("best score:", grid.best_score_)
print(grid.cv_results_)
