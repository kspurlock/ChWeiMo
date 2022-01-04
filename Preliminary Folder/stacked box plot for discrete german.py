import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import Range1d


def preprocessing():
    dataset = pd.read_csv('german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

dataset = preprocessing()
data = dataset.iloc[:, [1, 2, 3, 4, 5, 8]]
cats = data.columns.values.tolist()

values = np.arange(0, max(data.max())+1).tolist()
values_str = list(map(str, values))

data_counts = np.empty((1, 6), dtype="int")

"""
for name in cats:
    counts = []
    for val in values:
        mask = np.where(data[name] == val)
        count = np.count_nonzero(mask)
        counts.append(count)
    counts = np.array(counts)
    data_counts = np.vstack()
"""
for val in values:
    counts = []
    for name in cats:
        mask = np.where(data[name] == val)
        count = int(np.count_nonzero(mask))
        counts.append(count)
    count = np.array(counts).reshape(1,-1)
    data_counts = np.vstack((data_counts, count))

data_counts = np.delete(data_counts, 0, axis=0)
data_dict = {"Features":cats}
for i in range(data_counts.shape[0]):
    data_dict[values_str[i]] = data_counts[i].tolist()
    
#%%

output_file("stacked.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]
colors = ["#2f85e0", "#abcaeb", "#e384e1", "#05ab08", "#8f581e", "#114c8c", "#0a8a0c", "#83b3e6"]

data_fruit = {'fruits' : fruits,
        '2015'   : [2, 1, 4, 3, 2, 4],
        '2016'   : [5, 3, 4, 2, 4, 6],
        '2017'   : [3, 2, 4, 4, 5, 3]}

"""
0 (6, )
1 (6, )
2 (6, )
3 (6, )
...
7 : (6,)
"""
#%%
p = figure(x_range=cats, height=250, title="German Credit Risk - Discrete",
           toolbar_location=None, tools="hover", tooltips="$name @fruits: @$name")

p.vbar_stack(values_str, x='Features', width=0.9, color=colors, source=data_dict,
             legend_label=values_str)

p.y_range=Range1d(0, 750)
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

show(p)