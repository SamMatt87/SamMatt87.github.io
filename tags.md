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

Next I wrote a function to clean the text in the titles. This involved using BeautifulSoup to remove any html tags, using re to replace any character that wasn't a letter, making the entire text lower case, and removing any words that were in the nltk stopwords. I then removed any rows that were tagged with 'N/A' from the previous section.

```python
def to_words(raw_text):
	text = BeautifulSoup(raw_text, features = "html.parser").get_text()
	letters = re.sub("[^a-zA-Z]"," ",text)
	lower_case = letters.lower()
	words = lower_case.split()
	stops = set(stopwords.words('english'))
	meaningful_words = [w for w in words if not w in stops]
	return (",".join(meaningful_words))

tagged_reduced = tagged[tagged.Code != 'N/A']
print(tagged_reduced.head())
```

The data was then split into a training and testing set. The text cleaning function was then applied on the data to ease the vectorisation of the variable. A count vectorizer with a limit of 5000 words was run on the cleaned text. 

```python
x_train, x_test, y_train, y_test = train_test_split(tagged_reduced.Question,tagged_reduced.Code,test_size=0.2)
y_train = np.asarray(y_train)
num_qs = x_train.size 
print(x_train.values[1])
clean_q_train =[]
for i in range (0,num_qs):
	clean_q_train.append(to_words(x_train.values[i]))
	if i%1000 == 0:
		print("Question %d of %d\n"%(i+1,num_qs))
	elif i==(num_qs-1):
		print('finished')
vectorizer = CountVectorizer(analyzer='word', tokenizer = None, stop_words=None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_q_train)
#train_data_features = train_data_features.values()
```

The count of the occurences of each word was then returned. The cleaninjg function and count vectoriser were run on the test set to return the vectorisaton matrix.

```python
vocab = vectorizer.get_feature_names()
dist=np.sum(train_data_features, axis=0)
print(dist)
for i in range(0,len(vocab)):
	print(vocab[i],dist[0,i])
#print (train_data_features.shape)
#print(train_data_features)
#for i in range(0,num_qs):
	#print (train_data_features[i])
clean_q_test=[]
num_test_qs = x_test.size
for i in range (0,num_test_qs):
	clean_q_test.append(to_words(x_test.values[i]))
	if i%1000 == 0:
		print("Question %d of %d\n"%(i+1,num_test_qs))
	elif i==(num_test_qs-1):
		print('finished')
test_data_features = vectorizer.transform(clean_q_test)
#train_data_features = np.asarray(test_data_features)
#print(train_data_features.dtype)
```

Finally a support vector categorisation model was fit to the training data and run on the test data with the accuracy being returned as aroud 71%.

```python
clf = SVC(gamma='auto')
clf.fit(train_data_features,y_train)
y_predict = clf.predict(test_data_features)
score = accuracy_score(y_test,y_predict)
print(score)
```

## Improvements
I greatly reduced the number of both questions and tags in this project. In the future it may be better to use one hot encoding to have a column for each of the top 100 tags or more. I ran the clean text function on each line individually when it would have been more efficient to apply it to the entire column using a lambda function. This was still early in my journey into data science so I did not know as much about these techniques as I do now.

You can find the full code for this project online [here](https://github.com/SamMatt87/Data-science-sample-projects/blob/master/Tags/tags.py)