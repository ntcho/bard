import math
import copy
import random
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


def eval_SSE(k, centers, clusters):
    # add distance to their cluster's distance
    return sum(
        [
            int(
                sum([d(p[1:], centers[i, 1:]) for p in clusters[i]])
                / (len(clusters[i]) if len(clusters[i]) > 0 else 1)
            )
            for i in range(k)
        ]
    )


def eval_silhouette(k, centers, clusters):
    # silhouette coefficient s(i)
    s = []

    for i in range(k):
        if len(clusters[i]) > 0:
            a = sum([d(p[1:], centers[i, 1:]) for p in clusters[i]]) / len(clusters[i])

            min_dist = None
            min_dist_cluster = 0

            for j in range(k):
                if i is not j:
                    dist = d(centers[i, 1:], centers[j, 1:])
                    if min_dist == None or dist < min_dist:
                        min_dist = dist
                        min_dist_cluster = j

            b = sum(
                [d(p[1:], centers[min_dist_cluster, 1:]) for p in clusters[i]]
            ) / len(clusters[i])

            s.append((b - a) / max(a, b))

    return sum(s) / len(s)


def eval_purity(k, clusters):
    purity = []

    for i in range(k):
        if len(clusters[i]) > 0:
            labels = np.array(clusters[i])[:, 0]
            counts = np.bincount(labels)
            common_label_occurence = np.count_nonzero(labels == np.argmax(counts))
            # calculate purity
            purity.append(common_label_occurence / len(clusters[i]))

    return sum(purity) / len(purity)


# test K-Means with k
def k_means(k, train):
    title = f"Testing: Digit OCR with K-means clustering, where k={k}"
    print("\n" + title + "\n" + "-" * len(title))

    # matplotlib visualization
    fig, gs = None, None
    gs_width = 0

    if is_dev:
        fig = plt.figure(figsize=(9, 10), tight_layout=True)
        gs_width = math.ceil(k / 2)
        gs = gridspec.GridSpec(5, gs_width, hspace=0.75)
        plt.rcParams["font.family"] = "serif"
        plt.ion()
        plt.show()

    test_time = timer()

    # initialize cluster center positions
    centers = np.ndarray(shape=(k, len(train[0])))

    # randomly initialize center position for cluster k = 1
    centers[0] = random.choice(train)

    # randomly initialize k center for k > 2
    if k > 1:
        # K-Means++ initialization method
        for i in range(1, k):
            max_dist_point = train[0]
            max_dist_sum = -1
            for t in train:
                # calculate sum of distance from every center assigned
                dist_sum = sum([d(t[1:], c[1:]) for c in centers[0:i]])
                # assign point with max distance as next center
                if max_dist_sum < dist_sum:
                    max_dist_point = t
                    max_dist_sum = dist_sum
            centers[i] = max_dist_point

    print("Initialize complete [{:.3f}s elapsed]".format(timer() - test_time))
    test_time = timer()

    clusters = [[] for i in range(k)]
    prev_clusters = None

    iteration = 0

    eval_label = ["SSE", "Silhouette", "Purity"]
    eval = [[] for i in eval_label]

    # iterate clustering
    while True:
        # create a copy of current cluster for comparison
        prev_clusters = copy.deepcopy(clusters)
        clusters = [[] for i in range(k)]

        # assign points to cluster
        for t in train:
            min_dist_cluster = None
            min_dist = None

            for idx, c in enumerate(centers):
                dist = d(t[1:], c[1:])
                if min_dist_cluster is None or dist < min_dist:
                    min_dist_cluster = idx
                    min_dist = dist

            clusters[min_dist_cluster].append(t)

        # re-estimate center of clusters
        for i in range(k):
            if len(clusters[i]) > 0:
                centers[i] = np.sum(np.array(clusters[i]), axis=0) / len(clusters[i])

        iteration += 1
        print(
            "Clustering... (iteration={}) [{:.3f}s elapsed]".format(
                iteration, timer() - test_time
            )
        )
        test_time = timer()

        # evaluate current iteration
        eval[0].append(eval_SSE(k, centers, clusters))
        eval[1].append(eval_silhouette(k, centers, clusters))
        eval[2].append(eval_purity(k, clusters))

        print(
            "Evaluating... (iteration={}, SSE={:.2f}, silhouette={:.2f}, purity={:.2f}) [{:.3f}s elapsed]".format(
                iteration, eval[0][-1], eval[1][-1], eval[2][-1], timer() - test_time
            )
        )
        test_time = timer()

        # DEV: update plot every iteration
        if is_dev:
            # clear figure
            plt.clf()

            # set title
            # TODO: find ways to refresh only the gridspec, not the entire figure
            plt.suptitle(
                f"Digit OCR with K-means clustering, where k={k}", fontweight="bold"
            )

            # test image
            for i in range(k):
                x = i if i < int(k / 2) else i % int(k / 2)
                y = 0 if i < int(k / 2) else 1
                ax_center = fig.add_subplot(gs[y, x : (x + 1)])
                ax_center.set_title("Cluster #{}".format(i), fontsize=10)
                ax_center.imshow(arr2img(centers[i][1:]), cmap="gray")
                ax_center.xaxis.set_major_locator(ticker.NullLocator())
                ax_center.yaxis.set_major_locator(ticker.NullLocator())

            for i in range(len(eval)):
                ax = fig.add_subplot(gs[i + 2, :])
                ax.set_title(eval_label[i])
                ax.plot(range(iteration), eval[i])
                ax.set_xlabel("Number of iteration")
                ax.xaxis.get_major_locator().set_params(integer=True)

            plt.pause(0.00001)

        # check whether cluster stopped changing
        if [int(len(i) / 10) for i in clusters] == [
            int(len(i) / 10) for i in prev_clusters
        ]:
            # stop iteration if the cluster didn't change anymore
            # NOTE: should be [len(i) for i in clusters] == [len(i) for i in prev_clusters]
            # without dividing 10, but divide in 10 to stop earlier
            print(
                "Clustering complete (iteration={}) [{:.3f}s elapsed]".format(
                    iteration - 1, timer() - test_time
                )
            )
            test_time = timer()
            break

        if iteration == 50:
            break

    print("-" * len(title))
    print(f"Complete: Digit OCR with K-means clustering, where k={k}")
    print(f"- Clustered {len(train)} data points")
    print(f"- Converged in {iteration - 1} iterations")
    print("- Performance: (iteration={}, SSE={:.2f}, silhouette={:.2f}, purity={:.2f})".format(
            iteration, eval[0][-1], eval[1][-1], eval[2][-1]
    ))

    return clusters


prev_time = timer()

# environment setting; dev for verbose, non-dev for performance
is_dev = False
# uncomment below to enable plotting & profiling
is_dev = True

# profiling setting; true to use viztracer
is_profiling_enabled = False
# is_profiling_enabled = True

# hyperparameter k
k = 10

# DEV: modify below to use subset of training data
train_limit = None
# train_limit = 5000

# DEV: performance profiling tool
if is_profiling_enabled:
    from viztracer import VizTracer

    tracer = VizTracer(max_stack_depth=3)
    tracer.start()

# train set; test set will iterate over this and compare
# `dtype=np.uint8` for memory optimization, `engine="pyarrow"` for I/O performance
input_train = "mnist_train.csv"  # 100MB
train = pd.read_csv(
    input_train, delimiter=",", header=None, dtype=np.uint8, engine="pyarrow"
).values
train = train if train_limit is None else train[0:train_limit]

print("Completed reading CSV files. [{:.3f}s elapsed]".format(timer() - prev_time))

k_means(k, train)

print()
print("Completed K-means clustering. [{:.3f}s elapsed]".format(timer() - prev_time))

# DEV: stop profiling
if is_profiling_enabled:
    tracer.stop()
    tracer.save()

# DEV: wait before closing the plot window
if is_dev:
    input("Press enter to exit...")
