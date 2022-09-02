import numpy as np
import pandas as pd
import time
import sys

sys.path.append("../CHWEIMO")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from chweimo.counterfactual import Optimizer
from chweimo.utils import split_by_cm
from chweimo.explain import aggregate_cf, aggregate_weight

import os

os.environ["OMP_NUM_THREADS"] = "2"


def preprocessing():
    dataset = pd.read_csv("./tests/data/german_credit_data.csv").drop("Unnamed: 0", axis=1)

    dataset.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True)

    for column in dataset.columns:
        if dataset[column].dtype != np.dtype("int64"):
            dataset[column] = LabelEncoder().fit_transform(dataset[column])

    return dataset


def train_model(x, y, model):
    cv = KFold(n_splits=3, random_state=None)
    global total_cm
    total_cm = np.zeros((2, 2))
    metric_dict = {}
    split_dict = {}

    it = 0
    for train_ind, test_ind in cv.split(x):
        x_train, x_test = x[train_ind], x[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        total_cm += confusion_matrix(y_test, y_pred)

        metrics = [
            accuracy_score(y_test, y_pred) * 100,
            precision_score(y_test, y_pred) * 100,
            recall_score(y_test, y_pred) * 100,
        ]

        metrics = np.round(metrics, decimals=2)

        metric_dict[it] = metrics
        split_dict[it] = (train_ind, test_ind)
        it += 1

    return total_cm, metric_dict, split_dict


"""Begin main execution"""
if __name__ == "__main__":
    global input_shape

    dataset = preprocessing()

    model = RandomForestClassifier(n_estimators=100, criterion="entropy")
    # model2 = LogisticRegression(max_iter = 1000)

    cols = dataset.iloc[:, :-1].columns.values
    class_names = ["Bad Risk", "No Risk"]
    X = dataset.drop(dataset.columns[-1], axis=1).values
    Y = dataset.iloc[:, [-1]].values.reshape(-1,)

    cm, metrics, splits = train_model(X, Y, model)

    x_train, x_test = X[splits[0][0]], X[splits[0][1]]
    y_train, y_test = Y[splits[0][0]], Y[splits[0][1]]

    cm_splits, x_train, x_test, y_train, y_test = split_by_cm(X, Y, model)
    cm_labels = ["true_neg", "false_neg", "true_pos", "false_pos"]

    data_maximums = np.max(dataset.iloc[:, :-1])
    
    discrete_map = np.where(
        data_maximums < 20, 1, 0
    )  # Can use np.where(discrete_map == 1, cols, 0)
    continuous_map = np.where(data_maximums > 20, 1, 0)
    feature_map = {"continuous": continuous_map, "discrete": discrete_map}

    #%%
    start = time.process_time()  # Check how long non-plausible takes
    itera = 0
    cf_dicts = dict.fromkeys(cm_labels)
    
    plausible = True
    
    for section in cm_splits:  # Loop for each confusion matrix section
        section_F = []
        section_X = []
        section_X_orig = []
        section_pred = []

        explainer = Optimizer(X, Y, model)
        for sample in section:  # sample in each of the confusion matrix
            x_orig = sample
            x_orig_y = model.predict_proba(x_orig.reshape(1, -1)).reshape(-1)

            change_class = np.argmin(x_orig_y)  # what class do we need to change to

            res = explainer.optimize_instance(
                sample=x_orig,
                change_class=change_class,
                plausible=True,
                method="NSGA2",
                termination=1,
            )

            section_X.append(res.history[-1].pop.get("X"))
            section_F.append(res.history[-1].pop.get("F"))
            section_X_orig.append(sample)
            section_pred.append(change_class)

        """Once outside need to aggregate change weights and coefficients"""

        counterfactuals = aggregate_cf(
            section_X, section_F, section_X_orig, cm_labels[itera], cols, feature_map, plausible, "Credit Risk"
        )
        
        aggregate_weight(section_X, section_F, section_X_orig, cm_labels[itera], cols, plausible, "Credit Risk")
        
        cf_dicts[cm_labels[itera]] = counterfactuals
        itera += 1