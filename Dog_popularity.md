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

Many breeds require regular daily or weekly brushing to maintain a nice coat, others don't need to be brushed outside of the groomers. Some owners may see dogs that require brushing as high maintenance and therefore, they may not be as popular, others may see these sessions as bonding time or relaxing increasing the breeds popularity. The grooming frequency value used in this project rates the level of grooming required from 0.2 to 1 across five categories from `Ocassional Bath/Brush` to `Specialty/Professional`.

### Shedding

Dogs shed their fur, some breeds more than others. This can be a chore to clean up, especially if the dog gets up on the furniture. Shedding can also trigger allergies and hayfever so some sufferers will want as little shedding as possible. As such, dogs with a higher level of shedding may be less popular. The shedding value used in this project rates the level of shedding between 0.2 and 1 with five categories from `Infrequent` to `Frequent`.

### Energy Level

Different breeds require different levels of exercise. For some people, this may effect their busy lifestyle making more energetic dogs less popular. For others, this may be a motivation to get out more leading to more energetic dogs being more popular. The energy level value used in this project rates the energy level of breeds from 0.2 to 1 across five categories from `couch potato` (interestingly, the Basset Hound is the only breed in this category) to `Needs Lots of Activity`.

### Trainability

Some breeds are easier to train than others. Owners may just want to train the basics such as potty training or they may want to teach their dogs advanced tricks to show off to friends. Either way, the trainability level may effect a breeds popularity. The trainability value used in this project rates the trainabilty of the breed between 0.2 and 1 across five categories from `May be Stubborn` to `Eager to Please`.

### Demeanor

The reaction of different breeds to new people can vary. If a breed doesn't react well to new pople it may not be popular. Other breeds may get too excited around new people effecting their popularity. The demeanor value rates the friendliness of the breed towards new people between 0.2 and 1 across five categories from `Aloof/Wary` to `Outgoing`.

## Preprocessing


## Data Exploration


## The Model


## The Results