import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from timeit import default_timer as timer

# returns a formatted 28x28 array for imshow
def arr2img(arr):
    return np.split((255 - arr).astype(float) / 255, 28)


# returns euclidian distance between data points
def d(a, b):
    # should be np.sqrt(np.sum(...)), but omitted for performance
    # WARN: make sure to cast the data into correct data type as uint8 will overflow!
    return np.sum(np.square(a.astype(int) - b.astype(int)))


# test kNN with multiple k values
def kNN(k, test, train):
    title = f"Testing: Digit OCR with kNN, where k={k}"
    print("\n" + title + "\n" + "-" * len(title))

    max_k = max(k)
    unreadable = []  # image data of unreadable digits

    # matplotlib visualization
    fig, gs = None, None
    size_image = len(k)  # number of plots
    size_plot = max_k + 1  # number of images

    if is_dev:
        fig = plt.figure(figsize=(16, 9), dpi=80, tight_layout=True)
        gs = gridspec.GridSpec(
            3, size_image * size_plot, hspace=0.5, wspace=20
        )  # height = 3
        plt.rcParams["font.family"] = "serif"
        plt.ion()
        plt.show()

    # count current accuracy
    test_correct = [0 for i in range(len(k))]
    test_count = []
    test_accuracy = []
    for i in k:
        test_accuracy.append([])
    # number of misclassification for each digits (0, 1, ... 9)
    test_digit_accuracy = [[0 for i in range(10)] for i in range(len(k))]
    test_confusion_matrix = [
        [[0 for i in range(10)] for i in range(10)] for i in range(len(k))
    ]

    test_time = timer()

    # iterate through test set
    idx_te = 0
    for te in test:
        # create empty array size of the training set
        dist = np.empty(shape=(len(train), 3), dtype=int)

        # compare distance with every example in training set
        idx_tr = 0
        for tr in train:
            dist[idx_tr] = [
                d(tr[1:], te[1:]),  # distance to training example
                tr[0],              # label of training example
                idx_tr,             # index of training example
            ]
            idx_tr += 1

        # max(k)-th closest data points
        # sort with column 0 (distance), extract k rows
        y = dist[dist[:, 0].argsort()][:max_k]

        test_wrong_count = 0

        # test whether prediction is correct or not
        for i, ki in enumerate(k):
            # find most common label from kNN's labels
            values, counts = np.unique(y[:ki, 1], return_counts=True)
            out = values[np.argmax(counts)]

            # update accuracy metrics
            if te[0] == int(out):  # if test label is same as predicted label
                test_correct[i] += 1
            else:
                test_wrong_count += 1
                # add correct label to misclassification record
                test_digit_accuracy[i][int(te[0])] += 1
                print(
                    f"[X] Misclassification (k={ki}): test label={te[0]}, kNN labels={y[:ki, 1]} indexes={y[:ki, 2]}"
                )

            test_confusion_matrix[i][int(te[0])][
                int(out)
            ] += 1  # update confusion matrix
            test_accuracy[i].append((test_correct[i] / (idx_te + 1)) * 100)

        test_count.append(idx_te + 1)

        should_show_progress = len(test) < 100 or idx_te % (int(len(test) / 100)) == 0

        # no k could read this digit, store for further inspection
        if test_wrong_count == len(k):
            unreadable.append(te)

        # DEV: update plot when each 1% of test set is tested,
        # or when misclassification happens
        if is_dev and (should_show_progress or test_wrong_count > 0):
            # clear figure
            plt.clf()

            # set title
            # TODO: find ways to refresh only the gridspec, not the entire figure
            plt.suptitle(f"Digit OCR with kNN, where k={k}", fontweight="bold")

            # test image
            ax_test = fig.add_subplot(gs[0, 0:size_image])
            ax_test.imshow(arr2img(te[1:]), cmap="gray")
            ax_test.xaxis.set_major_locator(ticker.NullLocator())
            ax_test.yaxis.set_major_locator(ticker.NullLocator())
            ax_test.set_title("test")
            ax_test.set_xlabel("label={}".format(te[0]))

            # kNN images
            for idx, kNN_idx in enumerate(y[:, 2].astype(int)):
                ax_kNN = fig.add_subplot(
                    gs[0, (idx + 1) * size_image : (idx + 2) * size_image]
                )
                ax_kNN.imshow(arr2img(train[kNN_idx][1:]), cmap="gray")
                ax_kNN.xaxis.set_major_locator(ticker.NullLocator())
                ax_kNN.yaxis.set_major_locator(ticker.NullLocator())
                ax_kNN.set_title("{} NN".format(idx + 1))
                ax_kNN.set_xlabel("label={}".format(train[kNN_idx][0]))

            # average accuracy plot
            for i, ki in enumerate(k):
                ax_accuracy = fig.add_subplot(
                    gs[1, i * size_plot : (i + 1) * size_plot]
                )
                ax_accuracy.set_title(
                    "Average accuracy ({:.2f}%)".format(test_accuracy[i][-1])
                )
                ax_accuracy.plot(test_count, test_accuracy[i])
                ax_accuracy.set_yticks([80, 90, 100])
                ax_accuracy.set_xlabel("Number of tests")

                # digit misclassification plot
                ax_accuracy = fig.add_subplot(
                    gs[2, i * size_plot : (i + 1) * size_plot]
                )
                ax_accuracy.set_title("Number of misclassifications")
                ax_accuracy.bar(range(10), test_digit_accuracy[i])
                ax_accuracy.set_xlabel("Digit")
                ax_accuracy.set_xticks(range(10))
                ax_accuracy.set_yticks(range(10))

            plt.pause(0.001)

        # show progress when each 1% of test set is tested
        if should_show_progress:
            print(
                "Testing... (cases tested={}, average accuracy={:.2f}%) [{:.3f}s elapsed]".format(
                    idx_te, test_accuracy[0][-1], timer() - test_time
                )
            )
            test_time = timer()

        # DEV: pause on misclassification
        if is_logging_enabled and test_wrong_count > 0:
            input("[X] Misclassification, press enter to continue...")

        idx_te += 1

    # print training results to console
    for i, ki in enumerate(k):
        print("-" * 40)
        print(f"Complete: Digit OCR with kNN, where k={ki}")
        print(f"- Tested {len(test)} data points with {len(train)} training data points")
        print("- Average accuracy = {:.2f}%".format(test_accuracy[i][-1]))
        print(f"- Misclassifications per digit (0-9) = {test_digit_accuracy[i]}")
        print(f"- Confusion matrix (X=classified, Y=actual) = {test_confusion_matrix[i]}")

    print("-" * 40)
    print("- Unreadable digits written to mnist_test.unreadable.csv")
    # create a CSV file with unrecognizeable digits for further inspection
    with open("mnist_test.unreadable.csv", "w") as f:
        for n in unreadable:
            f.write(",".join(map(str, n.tolist())))
            f.write("\n")


prev_time = timer()

# environment setting; dev for verbose, non-dev for performance
is_dev = False
# uncomment below to enable plotting & profiling
is_dev = True

# logging setting; true to stop at every misclassification
is_logging_enabled = False
# is_logging_enabled = True

# profiling setting; true to use viztracer
is_profiling_enabled = False
# is_profiling_enabled = True

# hyperparameter k
k = [1, 3, 5, 7]

# DEV: modify below to use subset of test/training data
test_limit = None
# test_limit = 10
train_limit = None
# train_limit = 1000

# DEV: performance profiling tool
if is_profiling_enabled:
    from viztracer import VizTracer

    tracer = VizTracer()
    tracer.start()

# test set; the performance will be evaluated with this set
# `dtype=np.uint8` for memory optimization, `engine="pyarrow"` for I/O performance
input_test = "mnist_test.csv"  # 10MB
test = pd.read_csv(
    input_test, delimiter=",", header=None, dtype=np.uint8, engine="pyarrow"
).values
test = test if test_limit is None else test[0:test_limit]

# train set; test set will iterate over this and compare
input_train = "mnist_train.csv"  # 100MB
train = pd.read_csv(
    input_train, delimiter=",", header=None, dtype=np.uint8, engine="pyarrow"
).values
train = train if train_limit is None else train[0:train_limit]

print("Completed reading CSV files. [{:.3f}s elapsed]".format(timer() - prev_time))

kNN(k, test, train)
print()
print("Completed kNN testing. [{:.3f}s elapsed]".format(timer() - prev_time))

# DEV: stop profiling
if is_profiling_enabled:
    tracer.stop()
    tracer.save()

# DEV: wait before closing the plot window
if is_dev:
    input("Press enter to exit...")
