# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:46:32 2021

@author: kylei
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import os
import bokeh

def preprocessing():
    dataset = pd.read_csv('german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

dataset = preprocessing()
data = dataset.iloc[:, [6]]
cats = data.columns.values
"""
age = data.iloc[:,[0]].values
age = np.hstack((age, np.full(shape=(len(age), 1),fill_value="Age")))
"""

credit_amount = data.iloc[:,[0]].values
credit_amount = np.hstack((credit_amount, np.full(shape=(len(credit_amount), 1),fill_value="Credit amount")))

"""
duration = data.iloc[:,[1]].values
duration = np.hstack((duration, np.full(shape=(len(age), 1),fill_value="Duration")))
"""
data = np.vstack((credit_amount))
yy = np.array(data[:,[0]], dtype="float64").reshape(-1,).tolist()
g_np = data[:,[1]]
g = []
for i in range(len(g_np)):
    g.append(str(g_np[i][0]))

#%%
# generate some synthetic time series for six different categories
from bokeh.plotting import figure, show

# generate some synthetic time series for six different categories
df = pd.DataFrame(dict(score=yy, group=g))


# find the quartiles and IQR for each category
groups = df.groupby('group')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

# find the outliers for each category
def outliers(group):
    cat = group.name
    return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
out = groups.apply(outliers).dropna()
#%%
# prepare outlier data for plotting, we need coordinates for every outlier.
if not out.empty:
    outx = list(out.index.get_level_values(0))
    outy = list(out.values)

p = figure(title = "German Credit Risk - Continuous 2", tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

# stems
p.segment(cats, upper.score, cats, q3.score, line_color="black")
p.segment(cats, lower.score, cats, q1.score, line_color="black")

# boxes
p.vbar(cats, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
p.vbar(cats, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

# whiskers (almost-0 height rects simpler than segments)
p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

# outliers
if not out.empty:
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "white"
p.grid.grid_line_width = 2
p.xaxis.major_label_text_font_size="16px"

show(p)