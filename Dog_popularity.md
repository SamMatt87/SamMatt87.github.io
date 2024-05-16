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

## The Model


## The Results