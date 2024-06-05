---
layout: default
title: Dog Popularity
permalink: /dog_popularity
description: "Using XGBoost to classify dog breeds by popularity"
---
# Dog Popularity
## Background

Dogs can bring so much joy to peoples lives, I know mine does. Dogs also vary vastly betwwen breeds depending on size, temperament, exercise and grooming needs. Because of this variation, some breeds are more popular than others, but what is it that makes a dog popular. For this project, I am using data for the over 200  breeds registered by the American Kennel Club with an XGBoost model to attempt to find out what factors effect a breeds popularity.

## The Data

For this project, I am using a csv of data extracted from the American Kennel Club website available of Kaggle at https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset. This csv contains data for the over 200 breeds registered by the American Kennel Club. The csv has 21 columns for this project I will be focusing on 14 of these including:
- Temperament
- Popularity
- Min height
- Max height
- Min weight
- Max weight
- Min expectancy
- Max excpetancy
- Group
- Grooming frequncy value
- Shedding value
- Energy level value
- Trainability value
- Demeanor value

### Temperament

This column uses a number of terms to describe the breed separated by commas for example "Sweet", "Playful" or "Loyal". Some of these terms are likely to be strongly related so I will look into this during the exploration phase.

### Popularity

The American Kennel Club updates the popularity of different breeds every year. This is based on the number of dogs registered across America. This variable is a ranking with 1 representing the most popular breed. Though there are 277 breeds in this set, the ranking only goes up to 195, sugesting that there are a number of breeds that a ran was not collected for so these breeds will have to be removed from the data.

### Min/Max height and Min/Max weight

These variables measure the expected height and weight of the breed in centimetres and kilograms. Dogs are typically grouped into small medium and large breeds and some owneres may have a preference regarding size.

### Min/Max Expectancy

These variables measure the expected lifespan of the bred in years. Losing a dog can be tough on owners so owners who have experience with this may prefer dogs wiht longer lifespans.

### Group

The  American Kennel Club splits dogs into six separate groups based on what the breed was initially used for. These groups may be a way to find similar breeds whode popularity should be close together. The groups include:
- Herding
- Hound
- Non-sporting
- Toy
- Terrier
- Sporting
- Working

### Grooming frequency

Many breeds require regular daily or weekly brushing to maintain a nice coat, others don't need to be brushed outside of the groomers. Some owners may see dogs that require brushing as high maintenance and therefore, they may not be as popular, others may see these sessions as bonding time or relaxing increasing the breeds popularity. The grooming frequency category used in this project rates the level of grooming required across five categories from `Ocassional Bath/Brush` to `Specialty/Professional`.

### Shedding

Dogs shed their fur, some breeds more than others. This can be a chore to clean up, especially if the dog gets up on the furniture. Shedding can also trigger allergies and hayfever so some sufferers will want as little shedding as possible. As such, dogs with a higher level of shedding may be less popular. The shedding category used in this project rates the level of shedding between with five categories from `Infrequent` to `Frequent`.

### Energy Level

Different breeds require different levels of exercise. For some people, this may effect their busy lifestyle making more energetic dogs less popular. For others, this may be a motivation to get out more leading to more energetic dogs being more popular. The energy level category used in this project rates the energy level of breeds across five categories from `couch potato` (interestingly, the Basset Hound is the only breed in this category) to `Needs Lots of Activity`.

### Trainability

Some breeds are easier to train than others. Owners may just want to train the basics such as potty training or they may want to teach their dogs advanced tricks to show off to friends. Either way, the trainability level may effect a breeds popularity. The trainability category used in this project rates the trainabilty of the breed across five categories from `May be Stubborn` to `Eager to Please`.

### Demeanor

The reaction of different breeds to new people can vary. If a breed doesn't react well to new pople it may not be popular. Other breeds may get too excited around new people effecting their popularity. The demeanor category rates the friendliness of the breed towards new people across five categories from `Aloof/Wary` to `Outgoing`.

## Preprocessing

For this project, after the data was loaded, it went through a number of steps before being fed into the model. These steps included:
- Normalising the popularity rank
- Filtering columns
- Removing null values
- Splitting the temperament column
- One hot encoding the category columns
- Extract the target values

### Normalising the popularity rank

Normalisation is used in data processing to scale the distribution and make it easier for the model to output results. Given the uniform distribution of the ranks, we simply have to remove any null values, subtract the minimum ranks, then divide by the rank by the max rank to get a uniform distribution between 0 and 1. The code I used to do this can be seen below.

```
    data = pd.read_csv('akc-data-latest.csv')
    data = data[data['popularity'].apply(lambda x: str(x).isnumeric())]
    data['popularity'] = (data['popularity'].astype(int)-min(data['popularity'].astype(int)))/max(data['popularity'].astype(int))
```

### Filtering columns and removing null values

I do not need to use all the columns provided in this dataset as some of them share the same information in a different way. By providing the dataframe with the list of columns I want to keep, pandas is able to filter the dataframe for me. Once the columns have been filtered, I then remove any rows with null values as there is enough data available to ignore these values. The code I used for this is shown below.

```
    data_columns = ['name','popularity','temperament','min_height','max_height','min_weight','max_weight','min_expectancy','max_expectancy', 'group', 
        'grooming_frequency_category', 'shedding_category', 'energy_level_category', 'trainability_category', 'demeanor_category']
    data = data[data_columns]
    data = data.dropna()
```

### Splitting temperaments

The temperaments column is a list of traits separated by commas. I want to create one hot encoding style columns for each trait. To do this, I fed the data into a funciton called `split_temperaments`. This function first cycles through the temperament of each dog to create a complete list of available traits. By splitting each row on the commas, then applying strip to remove unnecessary white space, we can identify the individual traits. The list of traits is then sorted alphabetically and a new list replacing remaining spaces with underscores and adding the prefix `temperament_` is created for column names to easily identify these columns. A new list is then created to hold the attribute lists for each breed. The temperament column is then cycled through again to create a list of binaries for if each breed has a certain trait. This list of binaries is then appended to the main list of attributes mentioned earlier. The attributes list is then converted to a dataframe with a column for each of the 140 traits and using the columns list created earlier for titles. The new dataframe is then merged with the original using the dog name as a key and the combined datafame is returned to the main function. You can see the code for this below.

```
def split_temperaments(data):
    temperaments = []
    for dog in data.temperament:
        if pd.notna(dog):
            for attitude in dog.split(','):
                if attitude.strip() not in temperaments:
                    temperaments.append(attitude.strip())
    temperaments.sort()
    temperament_headings = [f"temperament_{temp.replace(' ', '_').lower()}" for temp in temperaments]
    temperaments_data = []
    for dog in data.temperament:
        if pd.notna(dog):
            attributes = [attribute.strip() for attribute in dog.split(",")]
        else:
            attributes = []
        breed_temperaments = []
        for temp in temperaments:
            breed_temperaments.append(temp in attributes)
        temperaments_data.append(breed_temperaments)
    attributes_df = pd.DataFrame(temperaments_data, columns = temperament_headings)
    attributes_df['name'] = data.name
    data_out = pd.merge(left=data, right=attributes_df)
    return data_out
```

### One Hot Encoding category columns

One Hot Encoding is a data processing technique that splits categorical columns into N (or N-1 depending on how nulls are handled) columns where N is the number of categories. The new columns are binary and have 1 where the category is true and 0 otherwise. Splitting a column this way makes the data easier to understand by the model, especially whent the categories may be numeric but not on a scale. It is good practice to do this even when the categories are not numeric. For this project, this was done for the following categories:
- Group
- Grooming fequency
- Shedding
- Energy level
- Trainability
- Demeanor

The pandas function `get_dummies` easily splits colums this way. I added a prefix to each column of `category_` and the category name to make the columns identifiable. As I had to do this for multiple columns I created the function below to make the process easily repeatable.

```
def one_hot_encoding(data, column):
    dummies = pd.get_dummies(data, columns=[column], prefix=[f'category_{column}'])
    return dummies
```

### Extract target values

The final step in the preprocessing was to extract the popularity target values and drop any columns no longer being used. The code for this is shown below.

```
    targets = variables['popularity']
    variables = variables.drop(['name','popularity'], axis=1)
    return data, variables, targets
```

## Data Exploration

Before building the model, I wanted to explore the data some more to gather insights and see how some of the variables might relate to each other. I generated a number of graphs including pie caharts, line graphs and matrices for a number of different variables and will go into greater detail in this section.

### Expectancy line graph

As mentioned earlier, some owners, especially those who have lost a dog in the past, may prefer a dog with a longer life expectancy. The data provided lists both a minimum and maximum life expetancy for each breed. To graph the results, I binned both of these categories for every full year using the pandas `value_counts` funcion, finding the number of dogs with a minimum or maximum expectancy for each year then using `sort_index` to get the years in order. I then plot both results on a single graph using the `right` value for each key to find the maximum of the binned range for the x value and the number of breeds for the y value. I then used numpy to create a range for the x labels and displayed a legend for each line. The plot was then saved to be referenced agian in the future. As I would be doing similar analyses for the breeds height and weight, I created the function `generate_line_graphs` for modularity. The code for this function can be seen below.

```
def generate_line_graphs(data, category, jump):
    top = int(max(data[f'max_{category}'])+1)
    min_results = data[data[f'min_{category}'].notnull()][f'min_{category}'].value_counts(bins = range(0,top, jump)).sort_index()
    max_results = data[data[f'max_{category}'].notnull()][f'max_{category}'].value_counts(bins = range(0,top, jump)).sort_index()
    plt.cla()
    plt.plot(max_results.keys().right, min_results.values, label=f'min_{category}')
    plt.plot(max_results.keys().right, max_results.values, label=f'max_{category}')
    plt.xticks(np.arange(0,top,jump))
    plt.legend()
    plt.savefig(f'{category}_plot.pdf')
generate_line_graphs(variables, 'expectancy', 1)
```

When looking at the output as shown below both the minimum and maximum expectancies appear to follow a bimodal distribution with the local maxima for the minimum expectancy at 10 and 12 years while the local maxima for the maximum life expectancy is at 12 and 15 years. We can also see that the minimums range from 5 to 15 years while the maximums range from 8 to 17 years. 

### Weight and height line graphs

Weight and height are used to measuere a dogs size. Apartments are more suited to smaller dogs while others may prefer a larger breed. Due to the large variation in sizes, I binned the height categories by 10 cm and the weight categories by 10 kg. I ran each of them through the previously mentioned `generate_line_graphs` function using the code below.

```
    generate_line_graphs(variables, 'height', 10)
    generate_line_graphs(variables, 'weight', 10)
```

From the weight plot, shown below, we can see what looks like a mix between and exponential and multimodal distribution with a spike at the 30kg mark for minimum and 40 kg mark for maximum. We can use these spikes to classify dogs into size categories with dogs with a minimum below 20kg as small/medium while dogs above 20 kg for minimum weight are large. This seems to reflect the general consensus among dog owners.



The height plot, shown below, follows a multimodal distribution with local maxima for minimum height at 30 and 60 cm tall and maximum height at 40 and 70 cm tall. Using these distributions, we can once again add a split to differentiate between small and large breeds at 40cm for the minimum height and 50cm for the maximum height.



### Category pie charts

I generated pie charts for each of the categories that I will split into one hot encoded columns to analyse their distrubution. As I was doing this for multiple columns I created the function `generate_pie` to perform this task. The function clears the plot with `cla`, generates the counts for each category using `value_counts` and plots them on a pie chart, then saves the plot to a file using `savefig` for future reference. This was performed on the raw data as it was easier before the one hot encoding split. The code for the function can be seen below.

```
def generate_pie(data, category):
    plt.cla()
    data[category].value_counts(dropna=True).plot.pie(title=category, ylabel='')
    plt.savefig(f'{category}_pie.pdf')
```

For group, there seems to be a fairly even spread across all the group categories as shown below. This uniform grouping could be helpfel to the model depending on where each breed ends up.



The most common grooming frequency category as shown in the chart below is `weekly brushing`. This is the second most intense brushing category. The catory on each side with regards to intensity `occasional brush/bath` and `3 timees a week brushing` are the next highest with `daily brushing` and `specialty/professional` only taking up a small percent each.



For the shedding categories, almost half the breeds fall into the `seasonal` category, the second lowest category. The `frequent` category that indicates the highest level of shedding only takes up a small percentage and the rest of the breeds are fairly uniformly split between the last three categories as shown below.



The vast majority of the breeds fall into the top three energy level categories with almost half being classified as `regular exercise`. This then decreases for the next two categories of `energetic` and even further for `needs lots of activity`. Only a small percentage fall under `calm` and `couch potato` which, as mentioned earlier, only contains one breed. You can see the chart below.



The two largest trainability categories are the two middle of the road categories of `agreeable` and `independent`. The percentage then decreases for `eager to please`, `easy training` and `may be stubborn`. You can see the chart for this below.



The most common demeanor categories are also the three middle ones `friendly`, `alert/reponsive` and `friendly`. You can see the chart below.




### Relationship Matrices

I wanted to see if there were any relationsips across two diferent variable types, the temperaments and across the categories such as group, grooming frequency, shedding, energy level, trainability and demeanor. To do this I created matrices to show the number of breeds in one category that appear in a second category both as a number and as an overall percentage of the number of second category instances. This involved creating two functions, `category_counts` to return the counts and percentages as numpy arrays and `generate_matrix` to convert these arrays to graphs. The category_counts function is fed the data and the string representing the start of the columns for the category. The data is filtered to have only these category columns then for each column we filter for rows where it has a value other than zero and sum up the values of the other columns. The column itself is set to zero so that the diagonal values are all zero. To find the percentage, these numbers are divided by the length of th filtered dataset. These two arrays are appended to their own lists and once all columns have been recorded, the lists are converted to arrays and returned along with the column names for labelling purposes. the code for `category_counts` is shown below.

```
def category_counts(data, category):
    category_data = data.filter(regex=f'{category}_')
    category_counts = []
    category_percentages = []
    for column in range(len(category_data.columns)):
        single_category = category_data[category_data[category_data.columns[column]]==True]
        category_count = single_category.sum().values
        category_count[column] = 0
        category_counts.append(category_count)
        if single_category.shape[0] == 0:
            category_percentages.append(np.zeros_like(category_count))
        else:
            category_percentages.append(category_count/single_category.shape[0])
    category_counts = np.asarray(category_counts)
    category_percentages = np.asarray(category_percentages)
    return category_counts, category_percentages, category_data.columns
```

The `generate_matrix` function converts these numpy arrays to graphs. It takes in the numpy array, the list of columns and a string for the filename to save the image to. I started by defining the ticks as a range from 0 to the number of columns which will be used later to make sure every category is represented in the labels. I then set the figure size and added a subplot to ensure the figure was large enough to see all the labels. The matshow function of pyplot is used to convert a 2D numpy array to a coloured matrix. I set the axis major locator to show every label and used the set_ticks functions to assign the columns to the ticks using the range I defined earlier rotating the x axis by 90 degrees so they didn't run into each other. I assigned a colorbar as a legend to the right side of the graph and saved the result to examine. You can see the code for this below.

```
def generate_matrix(data, columns, filename):
    ticks = range(0, len(columns))
    fig = plt.figure(figsize=(28,28))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(ticks, columns)
    ax.set_yticks(ticks, columns)
    ax.tick_params(axis='x', labelrotation=90)
    fig.colorbar(cax)
    plt.savefig(filename)
```

Starting with the demeanors going by overall count, you can see in the mtrix below that the main connection that sticks out is the combination of `loyal` and `affectionate` as these are the only two yellow squares in the matrrix, there is also the obvious combination of `smart` and `confident` as well as `curious` and `alert` among others.



More connections become visible when we look at percentages rather than overall counts as seen in the matrix below. We see many more yellow points where almost all the breeds with one quality are also given the other. Once more, we have obvious ones like `devoted` and `friendly` along with less obvious pairings like `low key` and `charming`.




Looking at the overall count of crossover between the six categories, as hown below, we can see a number of relationships that stand out. The most obvious link, being the only yellow points being the link between `weekly brushing` and `seasonal shedding` which makes sense. The `weekly brushing` grooming category also has some other links that stand out including with the `regular exercise` training category and the `friendly` demeanor.



More relationships come into view when we see this matrix by percentage rather than overall count. One example of a new link is between the `specialty/professional` grooming category and the `regular exercise` training category. This link may be a red herring though due to the small size of this grooming category. We also see yellow squares across the `couch potato` energy level category since, as mentioned earlier, there is only one breed in this category.



## The Model

For this model, I used an XGBoost Regressor. XGBoost or Extreme Gradient Boosting is an ensemble method where decision trees are generated in parallel using gradient boosting to improve upon previous generations. The first step of this model was to retrieve the data and split it into a train and test set. The data has already been provided by the code in the preprocessing section, then I used the `train_test_split` funciton from sklearn to split the data and the target outputs. I wanted to test multiple models with different parameters so I then constructed two dictionaries where the number of estimators would be the key and the values would be a list of error rates for each depth. Estimators in this case refers to the number of trees built, more trees would lead to more accurate results but at a certain point you may risk diminishing returns where the model takes longer to build due to the number of trees and the larger models only improve the accuracy by an insignificant amount. Depth refers to the maximum number of levels in the decision trees, like with the number of estimators, more levels can lead to greater accuracy on the training set but also results in more complex trees that take longer to build and may be overfit on the training data.

For the estimators, I wanted to test relatively low numbers against relatively high numbers so I ran code for estimators from 1 to 20 as well as the hundreds from 100 to 500. Cycling through these numbers and the max depth from 1 to 10, I ran each XGBoostRegressor model with the assigned estimators and depth and setting the device to `gpu` to speed up the process building the trees in parallel. I would then fit the model to the training data and save each as a json file using the number of estimators and depth in the filename. After using predict to generate the predictions for each set of data, I used the `mean` numpy function to fund the root mean squared error for each combination of estimators and depth. I recorded these results and saved them to graphs to find the ideal case of estimators and depth. Due to the risk of overfitting, I focused mainly on the test results rather than the train results. You can see the code and graphs for this section below.

```
def model():
    _, variables, targets = preprocess()
    train_variables, test_variables, train_targets, test_targets = train_test_split(variables, targets, test_size=0.2, random_state=77)
    train_errors = {}
    test_errors = {}
    for estimators in [#1,2,3,4,5,6,7,8,9,10,
                       #11,12,13,14,15,16,17,18,19,20,
                       100,200,300,400,500
                       ]:
        estimator_train_errors = []
        estimator_test_errors = []
        for depth in range(1,11):
            xgb = XGBRegressor(n_estimators = estimators, max_depth = depth, device = 'gpu')
            xgb.fit(train_variables, train_targets)
            xgb.save_model(f'estimators_{estimators}_depth_{depth}.json')
            train_predictions = xgb.predict(train_variables)
            predictions = xgb.predict(test_variables)
            train_rmse = np.mean(((train_targets-train_predictions)**2))**0.5
            rmse = np.mean(((test_targets-predictions)**2))**0.5
            estimator_train_errors.append(train_rmse)
            estimator_test_errors.append(rmse)
        train_errors[estimators] = estimator_train_errors
        test_errors[estimators] = estimator_test_errors
    for k, v in train_errors.items():
        plt.plot(range(1,11), v, label = k)
    plt.legend()
    plt.savefig("train_rmse_hundreds.pdf")
    plt.cla()
    for k, v in test_errors.items():
        plt.plot(range(1,11), v, label = k)
    plt.legend()
    plt.savefig("test_rmse_hundreds.pdf")
```



## The Results
From the graphs in the previous section, we can see that the best performing model is with 7 estimators at a max depth of 4. By loading the details in the json file for this model using `load_model` I can use `plot_tree` to visualise each of the 13 estimators. I saved these as pdfs with a high number of dots per inch to be able to examine each of these trees individually. You can see the code for this section below along with each of the trees.

```
from xgboost import XGBRegressor, plot_tree, plot_importance
from matplotlib import pyplot as plt

xgb = XGBRegressor()
xgb.load_model("estimators_7_depth_4.json")
for trees in range(0,13):
    plt.cla()
    plot_tree(xgb,num_trees= trees)
    plt.savefig(f'xgb_tree_{trees}.pdf', dpi = 300)
```



I then used the `plot_importance` function to find the importance of each variable by weight and gain. plotting the importance by weight shows us the importance based on the number of times a variable was used in the trees. By contrast, plotting by gain shows the importance by the average split whenever the variable is used. Both of these methods give important insights into the importance of each variable. I increased the figure size for these plots and moved the left side of the plot to accomodate for the length of some of the variable names. Due to the large number of variables, I only wanted to see the top 10 of each importance type so i set `max_num_features` to 10, I also set `height` to 1 to fill out the plots. You can see the code for this section and the resulting plots below.

```
figure = plt.figure(figsize=[20,10])
ax = figure.add_subplot(111)
figure.subplots_adjust(left=0.3)
plot_importance(xgb, height=1, ax=ax, max_num_features=10,importance_type="weight")
plt.savefig("weight_importance.pdf")

figure = plt.figure(figsize=[20,10])
ax = figure.add_subplot(111)
figure.subplots_adjust(left=0.3)
plot_importance(xgb, height=1, ax=ax, max_num_features=10,importance_type="gain")
plt.savefig("gain_importance.pdf")
```



Each of the plots above reveals a different set of important variables. For example, the plot based on weight shows that the minimum height being used 11 times across the trees followed by the maximum expectancy at 10 and the minimum weight at 9. The plot for the gain importance however leads with the independent training category followed by breeds in the hound group and breeds with a devoted temperament.

As a final insight, I wanted to compare the predicted dog popularity to the actual popularity for the top 10 dogs. To do this, I gathered the data from the preprocessing step and ran them through the model again to find their prdictions. I then created a function which multiplied the input by the number of breeds and added 1 to reverse the normalisation process using on the prediciton outputs and the target values. After that, I extracted the name for each breed from the data so that the reader would know the rank of each dog breed. I then combined the name, predicted output and initial rank into a dictionary that I converted into a dataframe. I sorted this dataframe by the prediction output and added a new column with a range starting from 1 to identify the order of the output. I also added a final column of the target rank minus the predicted rank to see the change in the rank postintions and printed the output to the console. You can see the code for this section and the output below.

```
data, variables, targets = preprocess()
predictions = xgb.predict(variables)
def reverse_normalistion(rank):
    rank = rank*192
    rank = rank + 1
    return rank

prediction_output = reverse_normalistion(predictions, data)
targets_rank = reverse_normalistion(targets, data)
names = data['name']
results_dict = {'name':names,'prediction_output': prediction_output, 'target_rank':targets_rank}
results_df = pd.DataFrame(results_dict, index=None)
results_df = results_df.sort_values(by=['prediction_output'], ignore_index=True)
results_df['predicted_rank'] = range(1,len(prediction_output)+1)
results_df['difference'] = results_df['target_rank'] - results_df['predicted_rank']
print(results_df.head(10))
```

```
                         name  prediction_output  target_rank  predicted_rank  difference
0           Yorkshire Terrier          31.959652         10.0               1         9.0
1                Poodle (Toy)          33.634834          7.0               2         5.0
2                     Bulldog          38.312851          5.0               3         2.0
3        Bernese Mountain Dog          38.792618         22.0               4        18.0
4  German Shorthaired Pointer          39.763603          9.0               5         4.0
5         German Shepherd Dog          39.763603          2.0               6        -4.0
6          Labrador Retriever          39.763603          1.0               7        -6.0
7              Boston Terrier          43.680702         21.0               8        13.0
8                    Brittany          45.223763         26.0               9        17.0
9                Newfoundland          46.284470         40.0              10        30.0
```

 Looking at the top 10, 6 of the 10 are still within the top 10 and the majority of the breeds (with the exception of the Newfoundland) moved less than 20 places which is about 10% of the total breeds. Using numpy to output statistics using `mean` and `median` as well as `abs` to convert the differences to all positve values, we find the mean of the differences is around 26.3 and the median is 20. The median being lower than the mean in this case suggests that most breeds have a difference score lower than the average and it is only a smaller number of breeds have a high difference dragging the average up.

 If you would like to learn more about this project, the repository is available at []
 
 ## [Return Home](https://sammatt87.github.io/)