"""
Employee Turnover Prediction

Goal        Predict the risk of an employee quitting
Dataset     https://www.kaggle.com/datasets/giripujar/hr-analytics
"""


import sys
from os.path import exists

# check whether the dataset CSV exists
if not exists("HR_comma_sep.csv"):
    print(
        'Error: Cannot find dataset at "./HR_comma_sep.csv". Download the dataset here: https://www.kaggle.com/datasets/giripujar/hr-analytics'
    )
    sys.exit(-1)

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
# import seaborn as sns


class HRDataset(torch.utils.data.Dataset):
    def __init__(self, file_name="./HR_comma_sep.csv"):
        """
        Part 1. Analysing Dataset

        14,999 rows, 10 columns of employee data
        - includes 23.8% of employee data who left

        Column names            Data type           Explanation
        -------------------------------------------------------------------------------------------------------------
        satisfaction_level      Decimal [0..1]      Current satisfaction score
        last_evaluation         Decimal [0..1]      Last performance evaluation score
        number_project          Integer             Number of projects assigned
        average_montly_hours    Integer             Number of hours spent working every month
        time_spend_company      Integer             Number of years worked at current company
        Work_accident           Integer [0 or 1]    0 if there were no accidents at work, 1 if not
        left                    Integer [0 or 1]    0 if employee is still working, 1 if not
        promotion_last_5years   Integer [0 or 1]    0 if employee hasn't received promotion in last 5 years, 1 if not
        Department              String              Name of department
        salary                  String              Amount of salary; "low", "medium", "high"
        """

        # import CSV
        self.data = pd.read_csv(file_name, encoding="utf-8")
        self._preprocess()

        # drop string based columns
        self.data = self.data.drop(["department", "salary"], axis=1)

        self.X = torch.from_numpy(
            StandardScaler().fit_transform(
                self.data.values[:, 1:]
            )  # normalize all features
        ).to(torch.float32)

        # label = first column; turnover
        self.y = torch.from_numpy(self.data.values[:, 0]).to(torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def _preprocess(self):
        """
        Part 2. Preprocessing Dataset

        - Rename column names
        - Move turnover (left) column to front (this will be the predicted label)
        - Remove duplicate rows
        - Vectorize string value columns (department, salary)   [denoted as V below]
        - Add 8 new features with feature extraction            [denoted as F below]

        Column names            Data type                   Explanation
        ----------------------------------------------------------------------------------------------------------------------
        turnover                Integer [0 or 1]            0 if employee is still working, 1 if not
        satisfaction            Decimal [0..1]              Current satisfaction score
        evaluation              Decimal [0..1]              Last performance evaluation score
        project_count           Integer                     Number of projects assigned
        hours                   Integer                     Number of hours spent working every month
        years                   Integer                     Number of years worked at current company
        had_accident            Integer [0 or 1]            0 if there were no accidents at work, 1 if not
        had_promotion           Integer [0 or 1]            0 if employee hasn't received promotion in last 5 years, 1 if not
        department              String *                    Name of department
        salary                  String *                    Amount of salary; "low", "medium", "high"
        department_id           Integer [0..10]             [V]
        salary_level            Integer [0, 1, 2]           [V]
        mean_deviation_of_project_count_by_years            [F]
        mean_deviation_of_project_count_by_department_id    [F]
        mean_deviation_of_hours_by_years                    [F]
        mean_deviation_of_hours_by_department_id            [F]
        mean_deviation_of_had_promotion_by_years            [F]
        mean_deviation_of_had_promotion_by_department_id    [F]
        mean_deviation_of_salary_level_by_years             [F]
        mean_deviation_of_salary_level_by_department_id     [F]

            * String values will be excluded from the training process
        """

        # rename column names to make it more readable
        self.data = self.data.rename(
            columns={
                "satisfaction_level": "satisfaction",
                "last_evaluation": "evaluation",
                "number_project": "project_count",
                "average_montly_hours": "hours",
                "time_spend_company": "years",
                "Work_accident": "had_accident",
                "promotion_last_5years": "had_promotion",
                "Department": "department",
                "left": "turnover",
            }
        )

        # move turnover to the first column
        turnover = self.data["turnover"]
        self.data.drop(labels=["turnover"], axis=1, inplace=True)
        self.data.insert(0, "turnover", turnover)

        # add "department_id" column
        self.data["department_id"] = pd.Series(
            pd.Categorical(self.data["department"], ordered=False)
        ).cat.codes  # unique interger id values for each department

        # add "salary_level" column
        self.data["salary_level"] = pd.Series(
            pd.Categorical(
                self.data["salary"], categories=["low", "medium", "high"], ordered=True
            )
        ).cat.codes  # converts salary to "low" = 0, "medium" = 1, "high" = 2

        # remove 3,008 duplicate rows
        self.data = self.data.drop_duplicates()

        # calculating correlation
        # print(self.data.corr())

        # add extracted features
        means = {}

        for name_a in [
            "project_count",
            "hours",
            "had_promotion",
            "salary_level",
        ]:
            means[name_a] = {}  # create empty dict

            for name_b in [
                "years",
                "department_id",
            ]:
                means[name_a][name_b] = {}  # create empty dict

                for val_b in self.data[name_b].unique():
                    mean_of_a_by_b = np.mean(
                        self.data.query(f"{name_b} == {val_b}")[name_a]
                    )
                    means[name_a][name_b][val_b] = mean_of_a_by_b

                # append new column with mean deviation values
                self.data["mean_deviation_of_" + name_a + "_by_" + name_b] = np.array(
                    [
                        val_a - means[name_a][name_b][val_b]
                        for val_a, val_b in zip(self.data[name_a], self.data[name_b])
                    ]
                )


"""
Part 3: Designing Neural Network Architectures
"""

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

# use custom dataset class
data = HRDataset()

# generate indices for training and testing rows
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    data.y,
    stratify=data.y,
    test_size=0.1,  # 10% of all data
    random_state=42,
)

batch_size = 64

# generate subset based on indices
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)

# create batches
train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=batch_size)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"

# loss function is binary cross entropy loss function with sigmoid function
loss_fn = nn.BCEWithLogitsLoss()


class NeuralNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(17, 128),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),  # hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 1),  # output layer
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x[:, 0]


model1 = NeuralNetwork1().to(device)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3)


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(17, 64),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(64, 32),  # hidden layer 2
            nn.ReLU(),
            nn.Linear(32, 16),  # hidden layer 3
            nn.ReLU(),
            nn.Linear(16, 1),  # output layer
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x[:, 0]


model2 = NeuralNetwork2().to(device)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)


class NeuralNetwork3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(17, 64),  # hidden layer 1
            nn.Tanh(),
            nn.Linear(64, 32),  # hidden layer 2
            nn.Tanh(),
            nn.Linear(32, 16),  # hidden layer 3
            nn.Tanh(),
            nn.Linear(16, 1),  # output layer
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x[:, 0]


model3 = NeuralNetwork3().to(device)
optimizer3 = torch.optim.SGD(model3.parameters(), lr=1e-3)


class NeuralNetwork4(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(17, 64),  # hidden layer 1
            nn.Tanh(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),  # hidden layer 2
            nn.Tanh(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),  # hidden layer 3
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),  # output layer
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x[:, 0]


model4 = NeuralNetwork4().to(device)
optimizer4 = torch.optim.SGD(model4.parameters(), lr=1e-3)

"""
Part 4. Training and Testing Neural Network Models
"""


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

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# metrics
accuracy = BinaryAccuracy()
precision = BinaryPrecision()
recall = BinaryRecall()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    metrics_loss = 0
    targets = torch.zeros([size])
    preds = torch.zeros([size])

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

    # save metrics for epoch progress reporting
    metric_log["accuracy"].append(metrics_accuracy.item())
    metric_log["precision"].append(metrics_precision.item())
    metric_log["recall"].append(metrics_recall.item())
    metric_log["loss"].append(metrics_loss)

    print(
        f"- Accuracy: {metrics_accuracy:>0.3f} \
        \n- Precision: {metrics_precision:>0.3f} \
        \n- Recall: {metrics_recall:>0.3f} \
        \n- Loss: {metrics_loss:>8f} \n"
    )


metric_log = {}
epochs = 750
current_epoch = 0

for model, optimizer in [
    (model1, optimizer1),
    (model2, optimizer2),
    (model3, optimizer3),
    (model4, optimizer4),
]:
    print("Using", model, "\n")
    metric_log = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "loss": [],
        "confusion_matrix": [],
    }

    for epoch in range(epochs):
        current_epoch = epoch
        print(f"[Training] Epoch {epoch+1}\n-----------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print()
        print(f"[Test] Epoch {epoch+1}\n-----------------------------")
        test(test_dataloader, model, loss_fn)

    print(f"[Results] {model.__class__.__name__}")
    print("Accuracy: ", "  ".join(f"{x:.3f}" for x in metric_log["accuracy"]))
    print("Precision:", "  ".join(f"{x:.3f}" for x in metric_log["precision"]))
    print("Recall:   ", "  ".join(f"{x:.3f}" for x in metric_log["recall"]))
    print("Loss:     ", "  ".join(f"{x:.3f}" for x in metric_log["loss"]))
    print()

    # save trained model for future use
    torch.save(model, f"{model.__class__.__name__}.pt")
