#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests as rq
import bs4
import re
from tqdm import tqdm
from time import sleep
from random import random


# In[2]:


PREFIX = "https://www.federalreserve.gov"

YEAR_LST2 = [f"{PREFIX}/newsevents/speech/{i}-speeches.htm" for i in range(2011, 2020)]
YEAR_LST1 = [f"{PREFIX}/newsevents/speech/{i}speech.htm" for i in range(2006, 2011)]

global log_failed_fed_reserve_fetch
log_failed_fed_reserve_fetch = []

def get_all(verbose=True):
    rst_lst = []

    itr = tqdm(YEAR_LST1 + YEAR_LST2) if verbose else YEAR_LST1 + YEAR_LST2
    for this_year in itr:
        response = rq.get(this_year)
        sp_obj = bs4.BeautifulSoup(response.text, "html.parser")
        all_article_obj_lst = sp_obj.find_all("a", {"href": re.compile("^/newsevents/speech/")})
        
        for this_article in all_article_obj_lst:
            sleep(random())
            
            
            this_href = this_article["href"]
            this_title = this_article.text
            this_date = re.findall("\d+", this_href)[0]
            this_content = rq.get(f"{PREFIX}{this_href}")

            print(this_date, this_title)
                       

            rst_lst.append((this_title, this_date,                            parse_one_article(this_content, title_= this_title, date_ = this_date)))
            
    if log_failed_fed_reserve_fetch:
        print('\n\n\n',"Failed fetch found. See log_failed_fed_reserve_fetch.\n")
            
    return rst_lst


def parse_one_article(response_obj, title_ = None, date_ = None):
    
    soup = bs4.BeautifulSoup(response_obj.text, 'html.parser')
    paras = soup.find_all('p')
    # find the index that real article starts
    start = 0
    for i in range(len(paras)):
        if "<p class" in str(paras[i]) and "<p class" in str(paras[i+1]) and "<p class" not in str(paras[i+2]):
            start = i+2
            break

    paras_ = paras[start:]
    
    #find the end of article
    end = 0

    end_signals = ["<p><strong>Footnotes", "<p>\n<strong>Footnotes","<p><strong>References",                    "<p>\n<strong>References","<p><a","<p>\n<a","<p><b>Footnotes","<p>\n<b>Footnotes",                  "<p><b>References","<p>\n<b>References"]

    for i in range(len(paras_)):
        for end_sig in end_signals:
            if (end_sig in str(paras_[i])):
                #print("stop signal activated:", end_sig)
                end = i
                break
        if end != 0:
            break

        if i>len(paras_)-2:
            end = i-1
            #print("boundary control activated")
            break

    article = ''
    for item in paras_[:end]:
        article = article + " " + item.text
    article = article[1:] # delete the first space
    
    
    if article == '':
        log_failed_fed_reserve_fetch.append((date_, title_))
        
    #print(article)    
    
    return article


# In[3]:


if __name__ == "__main__":
    rst = get_all()
    print(1)


# In[4]:


log_failed_fed_reserve_fetch


# In[ ]:




