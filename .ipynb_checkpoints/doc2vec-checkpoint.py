
# coding: utf-8

# In[7]:


from naive_bayes_classifier.load_data import str2word_bag
from naive_bayes_classifier.configure import *
from back_test import *
from utils import *
import pickle


# In[8]:


import gensim


# In[9]:


from functools import reduce


# In[10]:


rate_se, rst_lst = load_data()


# In[11]:


def str2article(orig_str):
    for c in STOP_CHARS:
        orig_str = orig_str.replace(c, "")
            
    wd_lst = orig_str.lower().split(" ")

    for swd in STOP_WORDS:
        try:
            wd_lst = list(filter(lambda x: x!= swd, wd_lst))
        except ValueError:
            pass

        
    return wd_lst 


# In[ ]:


#sen_by_atc_lst = [(i, str2sentence(ctt[2])) for i, ctt in enumerate(rst_lst)]
#sen_lst = reduce(lambda x, y: x + y, [ctt[1] for ctt in sen_by_atc_lst])

with open("atc_lst", "rb") as fp:
    atc_lst = pickle.load(fp)

# In[ ]:


def get_X_train(atc_lst):
    X_train = []
    for i, atc in enumerate(atc_lst):
        document = gensim.models.doc2vec.TaggedDocument(atc, tags=[i]) 
        X_train.append(document)
    return X_train


# In[ ]:


X_train = get_X_train(atc_lst)


# In[ ]:

try:
    model = gensim.models.doc2vec.Doc2Vec.load("d2v_no_train.pymdl")
except:
    model = gensim.models.doc2vec.Doc2Vec(X_train, min_coun=1, window=3, workers=16)
    model.save("d2v_no_train.pymdl")

# In[ ]:


tt = input("train?")
model.train(X_train, total_examples=model.corpus_count, epochs=500)
print("train done")

# In[ ]:


model.save("d2v.pymdl")

