import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassConfusionMatrix

# Based on https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Download training data from open datasets
training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets
test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")  # will be "cpu" on Apple Silicon

# Define model 1
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_block = nn.Sequential(
            nn.Linear(28 * 28, 128),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),  # hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 10),  # output layer
            nn.LogSoftmax(dim=1),  # softmax scaling
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_block(x)
        return x


# Basic NN model
model_basic = NeuralNetwork().to(device)
optimizer_basic = torch.optim.SGD(model_basic.parameters(), lr=1e-3)
# print(model_basic)

# Define model 2
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # hidden layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=2),  # hidden layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear_block = nn.Sequential(
            nn.Linear(64 * 6 * 6, 256),  # output layer
            nn.ReLU(),
            nn.Linear(256, 10),  # output layer
            nn.LogSoftmax(dim=1),  # softmax scaling
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # reshape to linear_block input dimension
        x = self.linear_block(x)
        return x


# CNN model
model_conv = ConvNeuralNetwork().to(device)
optimizer_conv = torch.optim.SGD(model_conv.parameters(), lr=1e-3)
# print(model_conv)

# loss function
loss_fn = nn.NLLLoss()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# metrics
accuracy = MulticlassAccuracy(num_classes=10)
precision = MulticlassPrecision(num_classes=10)
recall = MulticlassRecall(num_classes=10)
confusion_matrix = MulticlassConfusionMatrix(num_classes=10)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    metrics_loss = 0
    targets = torch.zeros([size])
    preds = torch.zeros([size, 10])

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            for i in range(len(y)):
                # save each batch results as array for metric computation
                targets[batch * 64 + i] = y[i]
                preds[batch * 64 + i] = pred[i]

            metrics_loss += loss_fn(pred, y).item()

    # compute metrics
    metrics_accuracy = accuracy(preds, targets)
    metrics_precision = precision(preds, targets)
    metrics_recall = recall(preds, targets)
    metrics_loss /= num_batches
    metrics_confusion_matrix = confusion_matrix(preds, targets)

    # save metrics for epoch progress reporting
    metric_log["accuracy"].append(metrics_accuracy.item())
    metric_log["precision"].append(metrics_precision.item())
    metric_log["recall"].append(metrics_recall.item())
    metric_log["loss"].append(metrics_loss)
    metric_log["confusion_matrix"].append(metrics_confusion_matrix)

    print(
        f"- Accuracy: {metrics_accuracy:>0.3f} \
        \n- Precision: {metrics_precision:>0.3f} \
        \n- Recall: {metrics_recall:>0.3f} \
        \n- Loss: {metrics_loss:>8f} \n"
    )


metric_log = {}
epochs = 20

for model, optimizer in [
    (model_basic, optimizer_basic),
    (model_conv, optimizer_conv),
]:
    print("Using", model, "\n")
    metric_log = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "loss": [],
        "confusion_matrix": [],
    }

    for t in range(epochs):
        print(f"[Training] Epoch {t+1}\n-----------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print()
        print(f"[Test] Epoch {t+1}\n-----------------------------")
        test(test_dataloader, model, loss_fn)

    print(f"[Results] {model.__class__.__name__}")
    print("Accuracy: ", "  ".join(f"{x:.3f}" for x in metric_log["accuracy"]))
    print("Precision:", "  ".join(f"{x:.3f}" for x in metric_log["precision"]))
    print("Recall:   ", "  ".join(f"{x:.3f}" for x in metric_log["recall"]))
    print("Loss:     ", "  ".join(f"{x:.3f}" for x in metric_log["loss"]))
    print("Confusion matrix:\n", metric_log["confusion_matrix"][-1])
    print()
