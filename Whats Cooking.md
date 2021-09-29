---
layout: default
title: What's Cooking
permalink: /SciKitLearn-basics/whats-cooking/
---
## Background
This is one of my early data science projects. For this project I wanted to practice creating tree models with SciKitLearn. This particular project looks at the ingredients in a dataset of recipes and attempts to predict the type of cuisine the recipe would be categorised under.

## The Dataset
The dataset I used for this project can be found on kaggle [here](https://www.kaggle.com/kaggle/recipe-ingredients-dataset). It came in the form of a json file that had the recipe id, the type of cuisine and a list of the ingredients. You can see a sample of the first three recipes below.
```json
[
  {
    "id": 10259,
    "cuisine": "greek",
    "ingredients": [
      "romaine lettuce",
      "black olives",
      "grape tomatoes",
      "garlic",
      "pepper",
      "purple onion",
      "seasoning",
      "garbanzo beans",
      "feta cheese crumbles"
    ]
  },
  {
    "id": 25693,
    "cuisine": "southern_us",
    "ingredients": [
      "plain flour",
      "ground pepper",
      "salt",
      "tomatoes",
      "ground black pepper",
      "thyme",
      "eggs",
      "green tomatoes",
      "yellow corn meal",
      "milk",
      "vegetable oil"
    ]
  },
  {
    "id": 20130,
    "cuisine": "filipino",
    "ingredients": [
      "eggs",
      "pepper",
      "salt",
      "mayonaise",
      "cooking oil",
      "green chilies",
      "grilled chicken breasts",
      "garlic powder",
      "yellow onion",
      "soy sauce",
      "butter",
      "chicken livers"
    ]
  }]
```

## The Process
The first step in this project was to import the necessary packages, this included pandas to import the data, numpy for any calculations involved and form the sklearn package train_test_split to create a test set to assess the model accuracy and tree to create a decision tree model.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
```

I then had to read the data from the json file and transform it into a data frame with one row for each ingredient in a recipe.

```python
print('reading data')
train = pd.read_json("train.json")
ingredient_series =[]
print('spliting by dish')
for i in range (0,len(train)):
	#print('dish ', str(i), ' of '+str(len(train)))
	for j in range(0,len(train['ingredients'][i])):
		ingredient_series.append([i,train['cuisine'][i],train['ingredients'][i][j]])
ingredients_list =pd.DataFrame(ingredient_series,columns=['dish no.','cuisine','ingredient'])
ingredients_list.to_csv('ingredients_list.csv')
```

After this, I created another data frame with a count of the frequency in which each ingredient occured in relation to the cuisine type.

```python
print('group by cuisine')
ingredient_counts_series =[]
for i in (np.unique(ingredients_list['cuisine'])):
	cuisine_ingredients = ingredients_list[ingredients_list['cuisine'] == i]
	for j in (np.unique(cuisine_ingredients['ingredient'])):
		#print('cuisine: ',i,' ingredient: ',j)
		ingredient_occurences = cuisine_ingredients[cuisine_ingredients['ingredient'] == j]
		ingredient_counts_series.append([i,j,ingredient_occurences['ingredient'].count()])
ingredient_counts = pd.DataFrame(ingredient_counts_series, columns=['cuisine','ingredient','count'])
ingredient_counts.to_csv('ingredient_counts.csv')
```

The next step was to split the first data frame I created into X and Y variables and split them so that 20% of the data became a test set to assess the model.

```python
print('training tree')
X = ingredients_list['ingredient']
Y = ingredients_list['cuisine']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
```

The CountVectorizer function was then imported from sklearn to create a labelled column for each type of ingredient.

```python
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
x_counts = CV.fit_transform(X_train)
x_counts_df = pd.DataFrame(x_counts.toarray())
x_counts_df.columns = CV.get_feature_names()
print(x_counts_df)
```

The tree was then trained on the training set with a max depth of 50 and the accuracy was tested against the test set.

```python
Groot = tree.DecisionTreeClassifier(max_depth = 50)
Groot.fit(x_counts_df,Y_train)
Groot.score(x_counts_df,Y_train)
print('calculating error')
x_counts_test_df = pd.DataFrame(CV.transform(X_test).toarray(),columns=CV.get_feature_names())
Groot_predict = Groot.predict(x_counts_test_df)
Groot_Error = 0
for i in range(0,len(Y_test)):
	if Y_test[i]!=Groot_predict[i]:
		Groot_Error += 1
print('Errors:',Groot_Error)
Accuracy = ((len(Y_test)-Groot_Error)/len(Y_test))*100
print('Accuracy:',Accuracy)
```

## Improvements
This project was still early days for me and I was still learning how to use most of the parameters of certain functions. Improvements based around this would include placing a limit on the number of features used by the count vectorizer and setting the max depth of the tree to much lower than 50 to avoid overfitting. On a project specific note, many ingredients such as all purpose flour are prevalent in many cuisines so could potentially be ignored, there are also ingredients that have different names but mean the same for example black pepper and ground black pepper. Such igredients could be grouped under the same label. Finally, sometimes it is the combination of ingredients rather than the individual ingredients themselves that make a recipe fit a specific cuisine and using the count vectorizer on individual ingredients rather than a list resulted in ingrediedients with more than one word being split into individual words. In the future I would not have a row for each ingredient of a recipe but a row for each recipe with all ingredients in a list.

You can view the full code for this project [here](https://github.com/SamMatt87/Data-science-sample-projects/blob/master/Whats%20cookin/whats%20cookin.py).