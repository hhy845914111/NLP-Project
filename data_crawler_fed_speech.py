
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 20, 10


# In[2]:


from naive_bayes_classifier.load_data import str2word_bag
from naive_bayes_classifier.configure import *


# In[3]:


from naive_bayes_classifier.back_test_utils import *


# In[4]:


from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score


# In[5]:


from sklearn.naive_bayes import MultinomialNB


# In[6]:


from sklearn.ensemble import ExtraTreesClassifier


# In[7]:


import pickle
import pandas as pd
import numpy as np


# In[8]:


with open("rst.pkl", "rb") as fp:
    rst_lst = pickle.load(fp)


# In[9]:


Y_FILE = "./USTREASURY-REALYIELD.csv"


# In[10]:


rate_se = pd.read_csv(Y_FILE, index_col=0)["10 YR"]
rate_se.index = pd.DatetimeIndex(rate_se.index)
rate_se = rate_se.sort_index()


# In[11]:


rate_se.plot();


# ## Overall reval profile 

# Reval/preval analysis is a widely used tools in market impact analysis. It is defined as the price movement before and after a certain incident happens in the market. In this case study, we first look at 5-year treasury yield movement before and after the Fed speech as a whole. Then we might want to generate features through the text and then to look at if the reval of differnet groups have diverged profiles.

# In[12]:


def get_reval(prev_n=10, rev_n=10):
    y_lst = []

    for ctt in rst_lst:
        prev = rate_se[rate_se.index <= pd.to_datetime(ctt[1])][-prev_n:].values
        rev = rate_se[rate_se.index > pd.to_datetime(ctt[1])][:rev_n].values
        this = prev[-1]
        
        tt = np.log(np.hstack((prev, rev)) / this)
        tt = np.where((np.abs(tt) == np.inf), 0.0, tt)
        tt = np.nan_to_num(tt)
        
        y_lst.append(tt)
    return y_lst


def evaluate(y_true, y_pred):
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average="weighted"), recall_score(y_true, y_pred, average="weighted"), f1_score(y_true, y_pred, average="weighted")



# In[13]:


rev_lst = get_reval()


# In[14]:


tt = np.vstack(rev_lst)


# In[15]:


plt.plot(tt.T);


# ## Roughly look at the distribution of revals

# In[16]:


import seaborn as sb


# In[17]:


sb.distplot(tt[:, 10]);


# In[18]:


sb.distplot(tt[:, 14]);


# In[19]:


sb.distplot(tt[:, 19]);


# In[20]:


y_raw = tt[:, 14]


# In[21]:


p = 0.2


# In[22]:


y_neg = (y_raw < np.quantile(y_raw, p)).astype(np.int)


# In[23]:


y_pos = (y_raw > np.quantile(y_raw, 1-p)).astype(np.int)


# In[24]:


np.sum(y_neg) / len(y_raw), np.sum(y_pos) / len(y_raw)


# In[25]:


benchmark_change = np.log(rate_se).diff().dropna().values


# In[26]:


y_benchmark_neg = (benchmark_change < np.quantile(y_raw, p)).astype(np.int)


# In[27]:


y_benchmark_pos = (benchmark_change > np.quantile(y_raw, 1-p)).astype(np.int)


# In[28]:


np.sum(y_benchmark_neg) / len(benchmark_change), np.sum(y_benchmark_pos) / len(benchmark_change)


# In[29]:


X_dct = [str2word_bag(itm[2], STOP_CHARS, STOP_WORDS, to_lower=True) for itm in rst_lst]


# In[30]:


X = pd.DataFrame(X_dct).fillna(0)


# In[ ]:


y = np.zeros_like(y_raw) - y_neg + y_pos


# In[ ]:


y_pred4 = rolling_test(ExtraTreesClassifier(), X.values, y, 100)


# In[ ]:


evaluate(y, y_pred4)


# In[ ]:


model = MultinomialNB()


# In[ ]:


model.fit(X, y)


# In[ ]:


y_pred = model.predict(X)


# In[ ]:



# In[ ]:


evaluate(y, y_pred)


# In[ ]:


plt.plot(np.cumsum(y_pred * y_raw))


# In[ ]:


model.feature_count_


# In[ ]:


prob = model.feature_log_prob_


# In[ ]:


prob.shape


# In[ ]:


prob_df = pd.DataFrame(np.exp(prob), index=[-1, 0, 1], columns=X.columns).T


# In[ ]:


dist = prob_df.std(axis=1).sort_values(ascending=False)


# In[ ]:


dist


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model2 = SVC()


# In[ ]:


model2.fit(X, y)


# In[ ]:


y_pred2 = model2.predict(X)


# In[ ]:


evaluate(y, y_pred2)


# In[ ]:


model3 = ExtraTreesClassifier()


# In[ ]:


model3.fit(X[:700], y[:700])


# In[ ]:


y_pred3 = model3.predict(X[700:])


# In[ ]:


evaluate(y[700:], y_pred3)


# ## TODO: Only use these as features.

# ## 1. Generate word bags using similar codes in naive_bayes_classifier

# ## 2. The reval characterisctic is captured by n day rate movement, we first set n == 1

# ## 3. Fit a naive bayes model to learn about which word/words could have the most impact of the rate.

# ## Future work
# 1. Use title information and treat title worlds differently as those in the body of the article
# 2. Cross-validation over time
# 3. Word2Vec
