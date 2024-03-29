---
layout: default
title: Lego sets classification
permalink: /lego_sets_classification
description: "Using the rebrickable API to classify Lego sets"
---

# Lego Sets Classificaiton
## Background

One of my hobbies is building Lego models. The website www.rebrickable.com has a database of Lego sets as well as a marketplace for people to buy and sell sets with others. They also have several  APIs to interact with this database which I will be using for this project. Due to rate limitations, it will take too long to collect sufficient data for every Lego set that has ever existed so I will be limiting this project to the "Star Wars" themed sets which will be celebrating their 25th anniversary in 2024. This theme currently has 168 sets and is targeted towards both children and adults so should be useful for this project. 

## API Interaction

I will be interacting with four different APIs for this project:
- `Theme API` to extract the name and number of each theme
- `Sets API`  to extract set details such as the release year and number of parts
- `Parts API` to extract the parts used in each set
- `MOCs API` to extract the alternate sets hobbyists have created using certain sets as a base

### Theme API

This API lets me find the name and reference number for each theme as well as other information such as the parent theme id allowing me to use other APIs to filter for sets in that theme. By searhing for the term `Star Wars` in the output, I should be able to find all the Star Wars themes. This was not the case however as one theme does not have Star Wars in the title but lists on of the star wars themes as its parent. To save vaving to loop through the themes more than once, I can assume that all parent themes appear before their children. I can then return a dictionary linking the theme numbers to the theme names as shown below.

```
def show_themes(name: str =None) -> Dict[int,str]:
    operation = "/api/v3/lego/themes"
    full_url = f"{url}{operation}"
    response = requests.get(full_url, headers= header)
    themes: Dict[int,str] = {}
    for entry in response.json()["results"]:
        if not name or name in entry['name'].lower() or entry['parent_id'] in themes.keys():
            themes[int(entry["id"])] =entry["name"]
    return themes
```
The output of this code looks like `{18: 'Star Wars', 158: 'Star Wars', 171: 'Ultimate Collector Series', 209: 'Star Wars'}`

### Sets API

Now that I have the themes I need to filter by, I can extract information for every set in that theme. The sets API allows me to search using theme, year range, number of parts range and search terms as well as order results by specific fields, It returns details such as the set number, name, release year and the number of parts. As many of these details are useful I return the data as is to extract individual details later. The code for this extraction is shown below.
```
def show_sets(theme: int, min_year: int = None, min_parts: int = None, ordering: str =None, search: str =None) -> List[Dict]:
    operation = "/api/v3/lego/sets"
    theme_id = f"theme_id={theme}"
    extras = ""
    if min_year:
        extras+=f"&min_year={min_year}"
    if min_parts:
        extras+=f"&min_parts={min_parts}"
    if ordering:
        extras+=f"&ordering={ordering}"
    if search:
        extras+=f"&search={search}"
    full_url = f"{url}{operation}/?{theme_id}{extras}"
    response = requests.get(full_url, headers=header)
    return response.json()["results"]

```
The output for a single set looks like this `[{'set_num': '75275-1', 'name': 'A-Wing Starfighter', 'year': 2020, 'theme_id': 171, 'num_parts': 1672, 'set_img_url': 'https://cdn.rebrickable.com/media/sets/75275-1/58298.jpg', 'set_url': 'https://rebrickable.com/sets/75275-1/a-wing-starfighter/', 'last_modified_dt': '2020-10-16T17:34:05.168530Z'}]`

### Parts API

The parts API lists the types of Lego parts in each set. This is useful for seeing the number of unique part types in each set. Some sets have more parts than would fit on a standard page so I needed to set a large page size to be sure to count all the unique parts. You can see the code for this extraction below.
```
def unique_parts(set: str) -> int:
    operation = f"/api/v3/lego/sets/{set}/parts/?page_size=1000"
    full_url = f"{url}{operation}"
    response = requests.get(full_url,headers=header)
    return len(response.json()['results'])
```

### MOCs API

In the Lego community a MOC or "My Own Creation" is a user-created set, sometimes based on the parts available in an existing set. This API returns the details of registered MOCs filtered by the base set that they started with. Looking at the number of MOCs based on each set may be an indication of the set classification. I included the code to extract this nuber below.
```
def moc_count(set: str) -> int:
    operation = f"/api/v3/lego/sets/{set}/alternates/?page_size=100"
    full_url = f"{url}{operation}"
    response = requests.get(full_url, headers=header)
    return len(response.json()['results'])
```

## Data Exploration

Using the APIs, I was able to visualise the spread of data across a number of variables I believed significant for categorising sets. These variables included:
- Release Year
- Number of parts
- Percentage of parts that are unique (number of part types/number of parts)
- Number of MOCs based on the set

### Release Year

This graph shows some interesting insights. First, we can see spikes for the release of The Phantom Menace in 2000 and Attack of the Clones in 2002/2003 but then there is a dip with little releases in 2005 for Revenge of the Sith. We then see a spike in 2009 with subsequent years matching it suggesting that this is due to the Clone Wars TV series. What's also interesting is that the buyout of Lucasfilm by Disney in late 2012 does not seem to have affected the number of sets released as they appear to remain in line with the years leading up to it. The sequel movies also do not appear to have associated spikes.

![Release year](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/96fb44fc-202c-430f-94bb-9dea677c8f05)


### Number of Parts

Looking at the number of parts, we can see two logarithmic-style curves. The first starts at sets with less than 100 pieces and ends in the 700 range, there is then a gap between this and when we start binning sets by thousands of pieces starting at the 1,000 range and again ending in the 7,000 range. This gap indicates that this variable could be very useful in splitting and classifying the sets.

![num pieces](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/32aceeed-68d0-46fa-829d-8561aa771a28)


### Unique Parts

This data also looks like a Bimodal distribution only this time of two normal distributions centered around 20% and 60%. Again, the fact that there are two peaks suggests a significant variable for our purposes. This analysis also bought to light that there are some sets that have 0 for their number of parts. This is obviously in error and these sets (5 of the total 168) will be excluded from the model.

![unique parts](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/54ef1985-7322-41da-b4a1-0c75d4b6e88d)


### MOCs

This is another logarithmic style curve with the vast majority of sets having no MOCs with them as the base. With this in mind, it may be better to have a flag for this variable in our model for whether or not any MOCs exist.

![num mocs](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/c8fd6f82-f86f-41ce-b5fc-c5be67bc9106)

## Modelling

The focus of this project is to demonstrate my understanding of unsupervised learning. The main difference between supervised and unsupervised learning is that for unsupervised learning, we do not include labels when training the model. This is because we are looking for new categorisations for the data that may not have been thought of before. Useful applications of this type of model include categorising types of customers by their buying patterns or, in the case of this project, categorising the target market for a Lego set using factors such as the number of pieces, the number of unique types and whether or not the online community has used this set to create their own. The type of model I will be using in this project is the K-Means model, this model splits the data into K clusters while trying to minimise the distance between points in the clusters and their mean position and stops once these two variables are stabilised. The model can take any number of input variables but is best visualised when using 2 or 3 variables. If we exclude the year of release variable we can see 3d representations of the clusters in the graph below with the number of clusters between 2 and 9 visualised by the number of parts. the percentage of unique parts and whether or not a MOC exists. 

![image](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/b2db7b10-441d-407e-95ca-aa30c15df53e)


By looking at the previous graphs, it can be seen that the existence of an MOC is not significant as it does not seem to be used in splitting the clusters. This can be seen further in the two sets of graphs below that show the MOC flag versus the number of pieces and the percentage of unique parts respectively.

![image](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/3b0f976e-f42b-4c66-8d00-07ce271ad0c3)

![image](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/2d0123d2-3e9f-46e1-8589-254997eb1893)

If we focus on only the number of pieces and the percentage of unique pieces instead, we can see better splits in the new graphs below.

![image](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/88f497e7-2f00-4b47-9dae-41a7205c62a6)


We can also plot these two variables against the release year, the one variable that we have not covered yet in the set of graphs below.

![image](https://github.com/SamMatt87/SamMatt87.github.io/assets/18587666/59e3c012-fc41-48e1-a36d-105092b2637a)

Looking at these graphs, I believe that the best representation of cluster comes from the number of pieces versus the percentage of unique parts splits and this is best split at 4 clusters. By looking at the sets, we can see that the cluster at the end includes the Milenium Falcon set, the AT-AT set, as well as the newly released Venator-Class Republic Attack Cruiser set. These sets are for more advanced users due to the complexity and the number of pieces. The next cluster includes multiple Death Star and Star Destroyer sets among others which seem the next step down in terms of complexity. We then have some simpler sets but still complex enough to be reserved for the 18+ range. The remainder of the sets are mainly for younger builders or collections of minifigures. Splitting the sets this way allowed us to separate the sets of different age ranges and complexities to determine their target market. It can also help to classify any new sets that may be added in the future.

If you would like to dive more into this analysis, the code files including the API extractor, the data exploration and a Jupyter notebook where I explore the different models are all available in the repository at [https://github.com/SamMatt87/Lego].

## [Return Home](https://sammatt87.github.io/)
