from sklearn.neighbors import KNeighborsClassifier


def kNN(k, test, train):
    x_train = train[:, 1:]
    y_train = train[:, 0]

    x_test = test[:, 1:]

    # train kNN model
    model = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=-1)
    model.fit(x_train, y_train)

    # test kNN model
    return model.predict(x_test)
