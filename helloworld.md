---
layout: default
title: Hello World
permalink: /SciKitLearn-basics/hello-world/
---
## Background
This was one of my first data science projects. I decided to run some models on the Titanic dataset considered by many to be the "Hello World" of data science. The goal of this project was to use details about the passengers to predict whether or not they survived the sinking of the Titanic.

## The Dataset
The dataset was downloaded from kaggle [here](https://www.kaggle.com/rahulsah06/titanic). It includes details such as the class they were travelling in, their gender, their age as well as other important variables including whether or not the passenger survied. A smple of the data can be seen below.

![image](https://user-images.githubusercontent.com/18587666/134831174-1f2d945f-7d81-429d-ab4b-ec90d01e222b.png)

## The Process
The first step in my code was to import the necessary packages. This included pandas for data extraction, train_test_split to split the data into treain and test sets, tree for the decision tree model, LabelEncoder to turn categorical variables into labels, numpy and math for mathematical operations on the data and grphviz to visualise the tree model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import numpy
import math
import graphviz
```