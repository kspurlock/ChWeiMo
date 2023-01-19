# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.append("../")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
# %%
X = pd.read_csv("./data/robot.csv").drop("Unnamed: 0", axis=1)
y = X["robustness"].values
X.drop("robustness", axis=1, inplace=True)

# %%
# Doing preprocessing on the column names now since the plot methods will reduce this to 5 chars
col_names = X.columns.values

for i, v in enumerate(col_names):
    a, b, c = col_names[i].split("_")
    a = b + c[:1]
    col_names[i] = a

class_names = ["Failure", "Success"]
X = X.values
# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=192)

# %%

# Aneseh's model
rf = RandomForestClassifier(n_estimators=100, max_features='log2', criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=123)

rf.fit(x_train, y_train.reshape(-1,))
yh = rf.predict(x_test)

print("Acc: ", accuracy_score(y_test, yh))
print("Prec: ", precision_score(y_test, yh))
print("Recall: ", recall_score(y_test, yh))


# %%
from chweimo.utils import split_by_cm

splits = split_by_cm(X, y, rf, test_size=0.10, plot_cm=True, class_names=class_names)

feature_map = {"continuous": np.ones((X.shape[1],)), "discrete": np.zeros((X.shape[1],))}
#%%
from chweimo.counterfactual import Optimizer

explainer = Optimizer(X, y, rf.predict_proba, col_names=col_names)
# %%
from chweimo.explain_tools import perform_aggregation
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

perform_aggregation(explainer, splits, type_dict=feature_map, data_name="Shadow Robot ", termination=100, barh=True, bar_label=False)

# %%
