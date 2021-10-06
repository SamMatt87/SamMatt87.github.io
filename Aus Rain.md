---
layout: default
title: Aus Rain
permalink: /SciKitLearn-basics/aus-rain/
---
## Background
This project uses the weather data for locations across Australia to predict whether or not it will rain there tomorrow. It uses different measurements of variables such as temperature, wind, air pressure, cloud cover and whether or not it rained today and feeds them into a decision tree model to make this prediction. It also outputs a visual representaion of the decision tree.

## The Dataset
THe dataset was sourced from kaggle [here](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package). It includes measurements of different weather related variables at the location at different points throughout the day including the minimum and maximum temperatures, the level of rainfall, the level of evaporation, the number of hours of sunshine, the direction and speed of the strongest wind throughout the day, the direction and speed of the wind at 9am and 3pm, the hunmidity at 9am and 3pm, the pressure at 9am and 3pm, the level of cloud cover at 9am and 3pm, the temperature at 9am and 3pm, wheter or not it rained today and whether or not it will rain tomorrow. You can see a sample of the data below.

![image](https://user-images.githubusercontent.com/18587666/136120538-52d2185b-368f-4a4d-aabf-ac7b2382e096.png)

## The Process
As usual, the first step ws to import some of the packages I would be using. These included numpy for performing calculations on the data, pandas to import and manipulate the data and from sklearn train_test_split to split the data into a train and test set, tree for buiding the decision tree model and accuracy_score for measuring the perfomance of the model.

```python
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
```

After importing the data I did some manipulation. This included binary categorisation for any yes or no variables, creating a month variable to represent the month of the year, changing any direction variables to the degrees from North and replacing null values with 0.

```python
data = pd.read_csv('weatherAUS.csv')
data=data.replace('No',0)
data=data.replace('Yes',1)
Month = pd.DatetimeIndex(data.Date).month.values
data['Month']=Month
print(data.Month)
direction = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
direction_numeric=[0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5]
for i in range(0,len(direction)):
	data = data.replace(direction[i],direction_numeric[i])
data = data.fillna(0)
print(data.head())
``` 

The next step was to create the X and Y datasets. For the X dataset I removed variables such as the location and date (I kept the month variable I created). The Y data was the target variable 'Rain Tomorrow'. I then used train test split to create a train and test set for each.

```python
x=data[['MinTemp','MaxTemp','Rainfall','Evaporation', 'Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','Month']]
y=data['RainTomorrow']
print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y)
```

I then built the model with a max depth of 20 and a minimum samples per leaf of 1,000. I fit the model on the train set, ran predictions on the test set and used accuracy score to return an accuracy of around 83%.

```python
clf = tree.DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 1000)
clf = clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
score = accuracy_score(y_test,y_predict)
print(score)
```

The final step was to visualise the tree I had created. I created a grphviz export of the model and used pydotplus to create the visualisation. I chose turquoise and orange for the category colours as they contrast each other well. I saved the tree to my local drive and you can see it below.

```python
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
import collections
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')
```

![tree](https://user-images.githubusercontent.com/18587666/136124069-3c5fd18f-78df-4294-bf79-0666dfe15fd0.png)


## Improvements
Some of the features that I changed to numeric variables like the month and the cardinal direction may have been improved by using one hot encoding instead. I also removed the location variable, dismissing it as not useful. However, some cities on the east coast of Australia receive more rain due to their proximity to both the coast and a mountain range. In the future, it may be useful to add variables like whether a location is on the coast or next to a mountain range to improve accuracy.

You can find the full code for this project online [here](https://github.com/SamMatt87/Data-science-sample-projects/blob/master/Aus_rain/Aus_rain.py)
