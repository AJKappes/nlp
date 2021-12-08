import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
# import string => string.punctuation

# url = 'https://www.mountainproject.com/route/105793664/outer-space'

service = Service('/home/alex/pyenv/chromedriver')
driver = webdriver.Chrome(service = service)
url = 'https://www.cnn.com/specials/opinion/opinion-politics'
driver.get(url)
html = driver.page_source.encode('utf-8') # grab raw html
html = html.decode('utf-8') # turn into text

#  find all urls
links = re.findall('<a href="(.+?)">', html)

# get opinion
date_pattern = r'^/[1-2]'
clean_op = set([op for op in links if re.search(date_pattern, op)])
op_links = [re.search(r'^.+?(?<=.com)', url).group(0) + op for op in clean_op]

# clean and get title, text
titles = []
text = []
clean_title_i = []
clean_text_i = []
for i in range(len(op_links)):
    
    op_page = urlopen(op_links[i])
    op_soup = BeautifulSoup(op_page, 'html.parser')
    
    try:
        titles.append(re.search(r'.*(?=\([O|o]pinion\))', op_soup.get_text()).group(0))
        print(str(i) + ' clean title')
        clean_title_i.append(i)
    
    except:
        titles.append('NotClean')
    
    try:
        text.append(re.search(r'(?<=\(CNN\)).*', op_soup.get_text()).group(0))
        print(str(i) + ' clean text')
        clean_text_i.append(i)
    
    except:
        text.append('NotClean')

clean_i = [i for i in clean_title_i if i in clean_text_i]
op_df = pd.DataFrame({'link': [op_links[i] for i in clean_i], 
                      'title': [titles[i] for i in clean_i],
                      'text': [text[i] for i in clean_i]})


# get polarity scores and bind dfs
sid = SentimentIntensityAnalyzer()
pols = [*op_df['text'].apply(sid.polarity_scores)]
pols_df = pd.DataFrame.from_records(pols)
scores_df = pd.concat([op_df, pols_df], axis=1)

# compound value [-1, 1] [neg, pos] 'sensationalism'
# threshold abs(2)
cutoff = .2
com_cond = [(scores_df['compound'] < -cutoff),
            (scores_df['compound'] > -cutoff) & (scores_df['compound'] < cutoff),
            (scores_df['compound'] > cutoff)]
vals = ['neg', 'neu', 'pos']
scores_df['com_lab'] = np.select(com_cond, vals)
scores_df['com_val'] = 0
scores_df.loc[scores_df['com_lab'] == 'pos', 'com_val'] = 1
scores_df.loc[scores_df['com_lab'] == 'neg', 'com_val'] = -1

# cleaning

stop_words = set(stopwords.words('english'))
op_token = word_tokenize(op_text)
op_filt = [w for w in op_token if w.casefold() not in stop_words
           and w in op_token if re.search(r'\w+', w)]

# vectorize
vectorizer = CountVectorizer()
X_op = vectorizer.fit_transform(op_filt)

sid = SentimentIntensityAnalyzer()

sid.polarity_scores(op_text)
