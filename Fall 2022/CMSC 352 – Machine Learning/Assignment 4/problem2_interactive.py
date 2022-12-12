"""
Employee Turnover Prediction Interactive Demo

Goal        Predict the risk of an employee quitting
Dataset     https://www.kaggle.com/datasets/giripujar/hr-analytics

* Try out index 2, 22, 2000, 6350, 9905!
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
        time_spend_company      Integer             Number of years at current company
        Work_accident           Integer [0 or 1]    0 if there were no accidents at work, 1 if not
        left                    Integer [0 or 1]    0 if employee is still working, 1 if not
        promotion_last_5years   Integer [0 or 1]    0 if employee hasn't received promotion in last 5 years, 1 if not
        Department              String              Name of department
        salary                  String              Amount of salary; "low", "medium", "high"
        """

        # import CSV
        self.data = pd.read_csv(file_name, encoding="utf-8")
        self._preprocess()

        # save full table for referencing
        self.full_table = self.data

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

    def __getrow__(self, index):
        return self.full_table.iloc[index]

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
        years                   Integer                     Number of years at current company
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

            nn.ReLU(),


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"


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


"""
Interactive Testing
"""

# use custom dataset class
data = HRDataset()

model_path = [
    "NeuralNetwork1.pt",
    "NeuralNetwork2.pt",
    "NeuralNetwork3.pt",
    "NeuralNetwork4.pt",
]


def predict_risk(model, X):
    # get prediction
    pred = model(
        torch.tensor(  # convert to 2d because original model is written for batch
            np.array(
                [
                    X.numpy(),
                ]
            )
        )
    )

    return 1 / (1 + np.exp(-pred.item()))  # sigmoid function


while True:
    print("\nEmployee Turnover Prediction with Neural Network")
    print("  [1] 2 hidden layer with ReLU")
    print("  [2] 3 hidden layer with ReLU")
    print("  [3] 3 hidden layer with Tanh")
    print("  [4] 3 hidden layer with Tanh and BatchNorm")
    print("  [0] Exit")

    try:
        model_id = int(input("Choose model to use: "))

        if model_id == 0:
            print("Exiting program.")
            sys.exit(0)  # exit program

        if model_id < 0 or model_id > 4:
            print("Error: Please enter number between 0-4.")
            continue

        if exists(model_path[model_id - 1]) == False:
            print(
                "Error: File for model doesn't exist. Please choose a different model."
            )
            continue

        # load trained pytorch model
        model = torch.load(model_path[model_id - 1])
        model.eval()
        
        # list of high risk employees
        # for i in range(len(data)):
        #     row = data.__getrow__(i)
        #     X, y = data.__getitem__(i)
        #     pred = predict_risk(model, X)
            
        #     if row["turnover"] == 0 and pred > 0.8:
        #         print(i, pred)

        while True:
            max_idx = len(data)
            try:
                print("\nData point selection")
                data_idx = int(input(f"Choose data point index [1-{max_idx}]: "))

                if data_idx == 0:
                    break

                if data_idx < 0 or data_idx > max_idx:
                    print(f"Error: Please enter number between 1-{max_idx}.")
                    continue

                row = data.__getrow__(data_idx - 1)
                print(f"Row {data_idx}:")

                if row['turnover'] == 1:
                    print("- Status: Previous employee")
                else:
                    print("- Status: Current employee")

                print(f"- Department: {row['department']}")
                print(f"- Salary level: {row['salary']}")
                print(f"  - Compared to same year employees: {row['mean_deviation_of_salary_level_by_years']:+.2f}")
                print(f"  - Compared to department average: {row['mean_deviation_of_salary_level_by_department_id']:+.2f}")
                print(f"- Satisfation rate: {int(row['satisfaction'] * 100)}%")
                print(f"- Evaluation score: {int(row['evaluation'] * 100)}%")
                print(f"- Projects assigned: {row['project_count']}")
                print(f"  - Compared to same year employees: {row['mean_deviation_of_project_count_by_years']:+.2f}")
                print(f"  - Compared to department average: {row['mean_deviation_of_project_count_by_department_id']:+.2f}")
                print(f"- Average monthly hours: {row['hours']}")
                print(f"  - Compared to same year employees: {row['mean_deviation_of_hours_by_years']:+.2f}")
                print(f"  - Compared to department average: {row['mean_deviation_of_hours_by_department_id']:+.2f}")
                print(f"- Years worked: {row['years']}")
                
                if row['had_accident'] == 1:
                    print("- Previously experienced accident")
                else:
                    print("- Never experienced accident")
                
                if row['had_promotion'] == 1:
                    print("- Received promotion in last 5 years")
                else:
                    print("- Didn't received promotion in last 5 years")
                print(f"  - Compared to same year employees: {row['mean_deviation_of_had_promotion_by_years']:+.2f}")
                print(f"  - Compared to department average: {row['mean_deviation_of_had_promotion_by_department_id']:+.2f}")
                    
                # print(row)

                X, y = data.__getitem__(data_idx - 1)

                risk = predict_risk(model, X)

                print()
                print(f"Predicted turnover risk: {risk * 100:.1f}%")
                
                # try out index 2, 22, 2000, 6350, 9905!

            except:
                print(f"Error: Please enter number between 1-{max_idx}.")

    except Exception as e:
        print("Error: Please enter number between 0-4.")
        print(repr(e))
