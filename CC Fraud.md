---
layout: default
title: CC Fraud
permalink: /SciKitLearn-basics/cc-fraud/
---
## Background
This project used a logistic regression function to pinpoint frauduent credit card transactions. It is important that we look at both the false positive rate and the true positive rate as in a real world scenario, fraudulent transactions only make up a small percentage of all transactions recorded.

## The Dataset
The dataset used for this project can be found on kaggle [here](https://www.kaggle.com/shubhamjoshi2130of/abstract-data-set-for-credit-card-fraud-detection). It includes fields such as the average transaction amount for the card, the amount for this transaction, whether the transaction was declined and whether it was a foreign transaction. You can see a sample of the data below.

![image](https://user-images.githubusercontent.com/18587666/136120239-47379f31-514d-48f5-82be-7006e6d5a8d2.png)
![image](https://user-images.githubusercontent.com/18587666/136120283-4d914010-e328-40ed-8d81-984bb0bc73d6.png)


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

Using these outputs, I was able to print statements returning the accuracy, log loss and the area under the curve.

```python
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
```

Finally, I plotted the ROC graph and found the sensitivity and specificity at the threshold point.

```python
idx = np.min(np.where(tpr > 0.95))
plt.figure()
plt.plot(fpr,tpr, color='coral', label = 'ROC Curve (area=%0.3f)'%auc(fpr,tpr))
plt.plot([0,1],[0,1],'k--')
plt.plot([0,fpr[idx]],[tpr[idx],tpr[idx]],'k--',color='blue')
plt.plot([fpr[idx],fpr[idx]],[0,tpr[idx]],'k--',color='blue')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate (1-specificity)', fontsize=14)
plt.ylabel('True positive rate (recall)', fontsize=14)
plt.legend(loc='lower right')
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
```

## Improvements
Some of the merchant IDs were repeated within the data so, rather than excluding them, I could have used them as extra columns. I also did not use all the statistical packages I imported and there was more analysis that could have been done on the resulting model.

You can see the full code [here](https://github.com/SamMatt87/Data-science-sample-projects/blob/master/CC%20Fraud/fraud.py)
