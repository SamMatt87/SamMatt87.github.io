---
layout: default
title: Tags
permalink: /SciKitLearn-basics/tags/
---
## Background
This project used natural language processing to create a support vector classification model which uses the title for a stack overflow question to predict the tags it relates to. Both the title of the post and the associated tags in list form are provided in the dataset.

## The Dataset
The dataset was sourced from kaggle [here](https://www.kaggle.com/badalgupta/stack-overflow-tag-prediction). The data contains 100,000 questions. There are two columns. The first column contains the title of the post on stack overflow as plain text. The second column contains the tags for the post, as a post can have multiple tags, the list of tags is contained within square parentheses, separated by commas and each tag is within quotation marks similar to how a list in python would appear when printed out. You can see a sample of the data below.

## The Process
Some python packages were first installed. The poackages included pandas to extract the data, numpy to perform calculations on the data, BeautifulSoup to remove any html that might appear in the titles, the regular expression package re to remove any punctuation from the titles, nltk to remove commonly used words from the titles and from the sklearn package train_test_split to create the training and testing sets, CountVectorizer to count the occurence of words in the title, SVC to create the model and accuracy_score to asses the model's accuracy.

```python
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

The next step was to import the data. I imported the data with pandas and ran some checks by returning the head of the dataframe and extracing the tags from the second row by takimg everything except the first and last character, then splitting on the commas and quote marks.

```python
data = pd.read_csv('train.csv')
print(data.head())
print(data['tags'][1][1:-1].split(',')[1].split("'")[1])
``` 

I then extracted the tags for each question in order to categorise the questions. I focused on four languages R, Python, Java and SQL. I created a new dataframe with the question and its tag.
```python
code = []
question = []
for i in range(0,len(data['tags'])):
	for j in range (0,len(data['tags'][i][1:-1].split(','))):
		x=data['tags'][i][1:-1].split(',')[j].split("'")[1]
		if x.lower() == 'r':
			code.append('R')
			question.append(data['title'][i])
		elif x.lower() == 'python':
			code.append('Python')
			question.append(data['title'][i])
		elif x.lower() == 'java':
			code.append('Java')
			question.append(data['title'][i])
		elif x.lower() == 'mysql':
			code.append('sql')
			question.append(data['title'][i])
		else:
			code.append('N/A')
			question.append(data['title'][i])
#print(code)
#print(question)
tagged = pd.DataFrame({'Question':question,'Code':code})
print(tagged.head())
```

