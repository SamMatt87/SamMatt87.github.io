---
layout: default
title: CC Fraud
permalink: /SciKitLearn-basics/cc-fraud/
---
## Background
This project used a logistic regression function to pinpoint frauduent credit card transactions. It is important that we look at both the false positive rate and the true positive rate as in a real world scenario, fraudulent transactions only make up a small percentage of all transactions recorded.

## The Dataset
The dataset used for this project can be found on kaggle [here](https://www.kaggle.com/shubhamjoshi2130of/abstract-data-set-for-credit-card-fraud-detection). It includes fields such as the average transaction amount for the card, the amount for this transaction, whether the transaction was declined and whether it was a foreign transaction. You can see a sample of the data below.

## The Process
The first step of the process was to import some python packages. I imported numpy for running calculatiions, pandas to import the data, pyplot from matplotlib to visualise the model and the logistic regression package from sklearn to build the model. I also imported the data and ran some checks by printing out the top 5 rows and counting the number of merchant ids.

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('creditcardcsvpresent.csv')
print(data.head())
print(data.Merchant_id.nunique())
```

I then imported train test split to split the data into train and test sets as well as cross val score that outputs the score for each stage of cross validation. I also imported a number of statistical metrics from sklearn including accuracy score, classification report, precision score, recall score, confusion matrix, precision recall curve, roc curve, auc and log loss. I then replaced any no values with 0 and yes values with 1 to binarise the categorical variables.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
data=data.replace('N',0)
data=data.replace('Y',1)
```

The next steps were to split the data into X and Y variable and create the train and test sets.

```python
X= data.iloc[:,2:11]
Y = data[['isFradulent']]

print(X)
print(Y.head())
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)
```

The model was then built on the training set and predictions were run on the test set. the false positive rate, true positive rate and threshold values were then calculated using the roc curve package.

```python
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
y_pred_proba = logreg.predict_proba(x_test)[:,1]
[fpr,tpr,thr] = roc_curve(y_test,y_pred_proba)
```
