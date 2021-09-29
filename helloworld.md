---
layout: default
title: Hello World
permalink: /SciKitLearn-basics/hello-world/
---
## Background
This was one of my first data science projects. I decided to run some models on the Titanic dataset considered by many to be the "Hello World" of data science. The goal of this project was to use details about the passengers to predict whether or not they survived the sinking of the Titanic.

## The Dataset
The dataset was downloaded from kaggle [here](https://www.kaggle.com/rahulsah06/titanic). It includes details such as the class they were travelling in, their gender, their age as well as other important variables including whether or not the passenger survied. A sample of the data can be seen below.

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

The first model type used to predict the outcome was a deciision tree. The only limitation I set on this tree was a maximum depth of 50. After running the fit and score methods on the training set, I compared the test set prediction to the actual values to assess model accuracy. I also used graphviz to show the nodes in the model. It had 45 errors for approximately 75% accuracy.

```python
Groot = tree.DecisionTreeClassifier(max_depth = 50)
Groot.fit(X_train,Y_train)
Groot.score(X_train,Y_train)
print('Decision Tree')
Groot_predict = Groot.predict(X_test)
Y_test_list = list(Y_test)
Groot_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=Groot_predict[i]:
		Groot_Error += 1
print('Errors:',Groot_Error)
Accuracy = ((len(Y_test)-Groot_Error)/len(Y_test))*100
print('Accuracy:',Accuracy)
groot_data = tree.export_graphviz(Groot, out_file ='out.dot', feature_names = X.columns, class_names = ['Survived','Dead'],filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(groot_data)
graph
```

The second type of model was a random forest. I set the number of estimators to 100, the maximum features to 5 and the maximum depth to 50. I fit the model on the training set and compared the results of the test set prediction to the real values for accuracy. This model had 33 errors for 81.5% accuracy.

```python
print('Random Forest')
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100, max_features = 5, max_depth = 50)
forest = forest.fit(X_train,Y_train)
RF_predict = forest.predict(X_test)
RF_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=RF_predict[i]:
		RF_Error += 1
print('Errors:',RF_Error)
RF_Accuracy = ((len(Y_test)-RF_Error)/len(Y_test))*100
print('Accuracy:',RF_Accuracy)
```

I then trained a logitic regression model. After fitting and scoring the model to the training data I printed out the intercept and coefficients to see how a change in each variable would effect the model. After this, I calculted the accuracy similar to the other models. This model had 42 errors for 76.5% accuracy.

```python
print('Linear Regression')
from sklearn.linear_model import LogisticRegression
Silverchair = LogisticRegression()
Silverchair.fit(X_train,Y_train)
Silverchair.score(X_train,Y_train)
print('Coefficient:\n',Silverchair.coef_)
print("Intercept:\n",Silverchair.intercept_)
Silverchair_predict = Silverchair.predict(X_test)
Silverchair_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=Silverchair_predict[i]:
		Silverchair_Error += 1
print('Errors: ',Silverchair_Error)
Silverchair_accuracy = ((len(Y_test)-Silverchair_Error)/len(Y_test))*100
print('Accuracy: ',Silverchair_accuracy)
```

Next was a support vector machine model. For this model I chose a linear kernel and set the nu to 0.4. As with the other models, I trained and scored it on the test data and calculated the accuracy. This model had 41 errors for 77% accuracy. 

```python
print('SVM')
from sklearn import svm
svm_model = svm.NuSVC(kernel = 'linear', nu = 0.4)
svm_model.fit(X_train,Y_train)
svm_model.score(X_train,Y_train)
svm_predict = svm_model.predict(X_test)
SVM_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=svm_predict[i]:
		SVM_Error += 1
print('Errors:',SVM_Error)
SVM_Accuracy = ((len(Y_test)-SVM_Error)/len(Y_test))*100
print('Accuracy:',SVM_Accuracy)
```

The final model I trined was a naive bayes. I set fit prior to false and alpha to 0.4. following a similar procedure to the other models to fit and score the model and retrieve the accuracy, the model had 62 errors for 65.3% accuracy.
```python
print('Naive Bayes')
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB(alpha = 0.4, fit_prior = False)
NB.fit(X_train,Y_train)
NB.score(X_train,Y_train)
NB_predict = NB.predict(X_test)
NB_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=NB_predict[i]:
		NB_Error += 1
print('Errors:',NB_Error)
NB_Accuracy = ((len(Y_test)-NB_Error)/len(Y_test))*100
print('Accuracy:',NB_Accuracy)
```

## Improvements
As I said at the start, this was one of the first data science projects I worked on. I didn't adjust many features for each model and only really looked at each models performance in their mos basic forms. I have since learned more about the adjustable parameters for each model type and are more familiar about the effect of adjusting each. I also am more familiar with techniques such as one hot encoding, which I would have used for the categorical variables.

You can view the full code for this project [here](https://github.com/SamMatt87/Data-science-sample-projects/blob/master/Hello%20World/hello%20world.py).