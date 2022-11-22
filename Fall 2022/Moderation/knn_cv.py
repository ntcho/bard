import cv2 as cv


def kNN(k, test, train):
    x_train = train[:, 1:].astype("float32")
    y_train = train[:, 0].astype("float32")

    x_test = test[:, 1:].astype("float32")

    # train kNN model
    model = cv.ml.KNearest_create()
    model.train(x_train, cv.ml.ROW_SAMPLE, y_train)

    # test kNN model
    retval, y_pred, neighbours, distance = model.findNearest(x_test, k)

    return y_pred
