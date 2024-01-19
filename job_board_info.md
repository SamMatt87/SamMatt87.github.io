---
layout: default
title: Job Board Information
permalink: /job_board_info
---
# Job Board Information
## Background
I want to be able to generate a customised resume for every job I apply for. I have built a tool, that I may share at a later date, to help with this process but it requires input such as:
- The name of the company to include in the document name
- The country of the job for wheteher I should include my phone number
- Keywords from the description to filter the skills I have that are required

This data is all available on the job ad page and is extractable via web scraping. Due to the different structure of each job board, the code to extract the data needs to be different for each. The main job boards I am using at the moment are:
- LinkedIn
- Seek

## LinkedIn
### Company Name
The easiest accessible company name is in a link tagged with a class name of `"topcard__org-name-link topcard__flavor--black-link"`. I used the `BeautifulSoup` package with `requests.get` to extract the text from this tag and removed any leading or trailing whitespace with `strip`. The full code can be seen below.
```
def get_company(url: str) -> str:
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    company_link = soup.find('a', class_="topcard__org-name-link topcard__flavor--black-link")
    company = company_link.get_text().strip()
    return company
```

### Country
The full location of the role including the city, state and country is stored in a span tag with the class name `"topcard__flavor topcard__flavor--bullet"`. Similarly to the company name, I used `Beautiful Soup` and `requests.get` to etract this text. Once I had the location, I searched the text for the keyword "Australia" to see if this was a local or international role. I would include my phone number if it was local but not if it was international as I am applying for both local and overseas roles. The code for this is shown below.
```
def is_Australian(url:str) -> bool:
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    Location = soup.find('span', class_="topcard__flavor topcard__flavor--bullet")
    Location_text = Location.get_text()
    if 'Australia' in Location_text:
        return True
    else:
        return False
```

### Role Description
I needed to find the skills mentioned in the role description to filter for the skills I should include in my resume. The role description covers an entire section of the page marked by a div tag with the class name `"description__text description__text--rich"` I ince again used `BeautifulSoup` and `requests.get` to extract the text of this section but as the section covered multiple tags, I needed to include the `separator` parameter to split the text from each tag and the `strip` parameter to strip the whitespace from each tag, rather than applying the `strip` function to the entire text as I had done previously, when extracting the text. I also want to remove the last 22 characters which is the text from the "Show More" and "Show Less" buttons. Once I have extracted the text, the next step is to cycle through keywords in my skills and see if they are listed in the description. This method is not perfect as it may highlight skills that are not needed such as highlighting "Management" skills when it sees the phrase "Database Management" and also ignores related keywords such as not highlighting my "Neural Networks" skills when the tool "pytorch" is mentioned. Both of these issues can be fixed with future updates and as I still manually check the skills required before generating the resume, this is a good first step. The code for the skills generation is shown below.
```
def skill_search(url: str, skills: List[str]) -> List[str]:
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    description = soup.find('div', class_="description__text description__text--rich")
    description_text = description.get_text(separator=". ", strip=True)[:-22]
    recomended_skills: List[str] = []
    for skill in skills:
        if skill.lower() in description_text.lower():
            recomended_skills.append(skill)
    return recomended_skills
```

## Seek
### Company
The  easiest way to extract the company name from Seek is by using the span tag with the `data-automation` attribute of `advertiser-name`. As there is no extra whitespace, I did not have to use `strip` when extracting the text using `BeautifulSoup` and `requests.get` as I did with LinkedIn. The code for this extraction is shown below.
```
def get_company(url:str) -> str:
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    company_link = soup.find('span', attrs={'data-automation':'advertiser-name'})
    company = company_link.get_text()
    return company
```

### Country
As seek is an Australian only website, I did not need to extract the country and set the defaul to True in regards to including my phone number.

### Role Description
Extracting the description was similar to the process for LinkedIn. The div section in this case had a `data-automation` tag of `jobAdDetails`. I had to include a separator, but not the split attribute when extracting the details with `BeautifulSoup` and `requests.get`. Afterwards, the same search method is used as with LinkedIn to find relevant skills with the same shortcomings. The code for this is shown below.
```
def skill_search(url:str, skills:List[str]) -> List[str]:
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    description = soup.find('div', attrs={'data-automation':'jobAdDetails'})
    description_text = description.get_text(separator=". ")
    recomended_skills:List[str] = []
    for skill in skills:
        if skill in description_text:
            recomended_skills.append(skill)
    return recomended_skills
```

## [Return Home](https://sammatt87.github.io/)