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

Next was the importing and preprocessing of the data. I imported the data from the csv, then created the X and Y datasets dropping the survived and any unique variables from the X data and only keeping survived for the Y data. I then ran two label encoders on the gender and embarked (which port the passengers joined the ship) variables. I also filled in the average age for any passengers where the age was missing. The data was then split into train and test sets in order to test the performance of the various models.

```python
passengers = pd.read_csv('train.csv')
X=passengers.drop(['PassengerId','Survived','Name','Cabin','Ticket',],axis=1)
Y=passengers['Survived']
Label_Encoder = LabelEncoder()
Label_Encoder2 = LabelEncoder()
gender_encoded = Label_Encoder.fit_transform(X['Sex'])
port_encoded = Label_Encoder2.fit_transform(X['Embarked'])
X['Sex'] = gender_encoded
X['Embarked'] = port_encoded
X['Age']=X['Age'].fillna(X['Age'].mean())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
```