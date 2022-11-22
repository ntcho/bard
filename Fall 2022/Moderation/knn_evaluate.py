from sklearn import metrics
import matplotlib.pyplot as plt


def evaluate(model_name, y_true, y_pred, visualize=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average="macro")
    recall = metrics.recall_score(y_true, y_pred, average="macro")
    f1 = metrics.f1_score(y_true, y_pred, average="macro")
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    print(f"[{model_name}] Performance analysis:")
    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1: {f1:.3f}")
    print("confusion matrix:", "\n", confusion_matrix)

    if visualize:
        confusion_matrix_plot = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=[str(i) for i in range(10)],
        )

        confusion_matrix_plot.plot(cmap="binary")

        plt.rcParams["font.family"] = "serif"
        plt.suptitle(f"{model_name} model", size="x-large", weight="bold")
        plt.title(f"Accuracy = {accuracy:.3f}", fontdict={"fontsize": 10})
        plt.show()
