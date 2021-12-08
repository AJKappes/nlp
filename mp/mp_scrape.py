import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

service = Service('/home/alex/pyenv/chromedriver')
driver = webdriver.Chrome(service = service)

# get WA route urls and setup storage

url = 'https://www.mountainproject.com/area/classics/105708966/washington'
driver.get(url)
time.sleep(3)
html = driver.page_source.encode('utf-8') # grab raw html
html = html.decode('utf-8') # turn into text

links = re.findall(r'<a href="(.*?)">', html)
clean_links = [r for r in links if not re.search(r'" class="', r)]
route_links = [r for r in clean_links if re.search(r'route/', r)]

route_keys = [re.sub(r'^.*?[0-9]\d+/', '', r) for r in route_links]
route_dict = {}
for r in route_keys:
    route_dict[r] = []

# helper funs - number comments and comment xpaths

def n_com():
    
    c_xpath = '/html/body/div[7]/div/div[3]/div/div[3]/div[3]/div/div[1]/h2'
    c_str = driver.find_element(By.XPATH, c_xpath).text
    n = int(re.findall(r'[0-9]{1,}', c_str)[0])
    
    return {'n': n , 'cxpath': c_xpath}

def xpath_comfun(i):
    
    p = '/html/body/div[7]/div/div[3]/div/div[3]/div[3]/div/div[2]/div[2]/table['+str(i)+']/tbody/tr/td[2]/div[2]/span[1]'
    
    return driver.find_element(By.XPATH, p).text

# scrape iteration

z = 1
for route in route_keys:
    
    print(str(z) + ' of ' + str(len(route_keys)) + ' routes')
    print('getting ' + route + '\n')
    r_url = [r for r in route_links if re.search(route, r)][0]
    
    driver.get(r_url)
    time.sleep(3)
    
    com_xpath = driver.find_element(By.XPATH, n_com()['cxpath'])
    driver.execute_script('arguments[0].scrollIntoView(true);', com_xpath)
    time.sleep(3)
    
    html = driver.page_source.encode('utf-8')
    html = html.decode('utf-8')
    
    comments = []
    for i in range(1, n_com()['n'] + 1):
        print(str(i) + ' comment[s] scraped')
        comments.append(xpath_comfun(i))
    
    route_dict[route] = comments
    print(route + ' complete\n')
    z += 1

# need df storage for each routes comments

####
driver.quit()

