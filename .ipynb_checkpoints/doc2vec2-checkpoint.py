
# coding: utf-8

# In[1]:


from naive_bayes_classifier.load_data import str2word_bag
from naive_bayes_classifier.configure import *
from back_test import *
from utils import *
from tqdm import tqdm


# In[2]:


import gensim


# In[3]:


from functools import reduce
from itertools import groupby


# In[4]:


rate_se, rst_lst = load_data()


# In[5]:


def str2sentence(orig_str):
    sen_lst = orig_str.split(".")
    
    rst_lst = []
    for sentence in sen_lst:
        wd_lst = sentence.lower().split(" ")
        
        for c in STOP_CHARS:
            sentence = sentence.replace(c, "")
        
        for swd in STOP_WORDS:
            try:
                wd_lst = list(filter(lambda x: x!= swd, wd_lst))
            except ValueError:
                pass
            
        sentence2 = " ".join(wd_lst)
            
        if sentence2 != "":
            rst_lst.append(sentence2.strip())
    
    return rst_lst


# sen_by_atc_lst = [(i, str2sentence(ctt[2])) for i, ctt in enumerate(rst_lst)]
# sen_lst = reduce(lambda x, y: x + y, [ctt[1] for ctt in sen_by_atc_lst])

# In[6]:


import pickle


# with open("sen_lst", "wb") as fp:
#     pickle.dump(sen_lst, fp)
#     
# with open("sen_by_atc_lst", "wb") as fp:
#     pickle.dump(sen_by_atc_lst, fp)

# In[7]:


with open("sen_lst", "rb") as fp:
    sen_lst = pickle.load(fp)
    
with open("sen_by_atc_lst", "rb") as fp:
    sen_by_atc_lst = pickle.load(fp)


# In[8]:


def get_X_train(sen_lst):
    X_train = []
    for i, sentence in enumerate(sen_lst):
        word_lst = sentence.split(" ")
        document = gensim.models.doc2vec.TaggedDocument(word_lst, tags=[i]) 
        X_train.append(document)
    return X_train


# In[9]:


X_train = get_X_train(sen_lst)


# model = gensim.models.doc2vec.Doc2Vec(X_train, min_coun=1, window=3, workers=16)

# model = gensim.models.doc2vec.Doc2Vec.load("d2v.pymdl")

# X_mat = np.zeros((len(sen_by_atc_lst), 100))
# 
# for atc_tpl in tqdm(sen_by_atc_lst):
#     tmp_lst = []
#     
#     for sentence in atc_tpl[1]:
#         try:
#             tmp_lst.append(model.infer_vector(sentence.split(" ")))
#         except TypeError:
#             pass
#     try:
#         X_mat[atc_tpl[0], :] = np.mean(np.vstack(tmp_lst), axis=0)
#     except:
#         X_mat[atc_tpl[0], :] = np.zeros(100,)

# np.save("X_mat", X_mat)

# X_mat.shape

# In[10]:


with open("atc_lst", "rb") as fp:
    atc_lst = pickle.load(fp)


# In[11]:


model = gensim.models.doc2vec.Doc2Vec.load("d2v.pymdl")


# In[ ]:


def doc2vec_backtest(rst_lst, embedding_model=gensim.models.doc2vec.Doc2Vec.load("d2v_no_train.pymdl"), batch_count=10):
    n_samples = len(rst_lst) // batch_count    
    
    X_train_vec_lst = []
    X_test_vec_lst = []
    
    for b_id in tqdm(range(batch_count - 2)):
        print("create X")
        this_X_train = [get_X_train(str2sentence(ctt[2])) for ctt in rst_lst[b_id*n_samples : (b_id+1)*n_samples]]
        print("X created")
        
        print("training")
        model.train(reduce(lambda x, y: x+y, this_X_train), total_examples=n_samples, epochs=500)
        print("train done")
    
        # get train vec rep
        print("predicting insample")
        this_X_train_vec = np.zeros((len(this_X_train), 100))
        for a_id, atc in enumerate(this_X_train):
            this_atc_vec = np.zeros((1, 100))
            
            for sentence in atc:
                this_atc_vec += model.infer_vector(sentence[0])
            
            this_X_train_vec[a_id, :] = this_atc_vec / len(atc)
        
        X_train_vec_lst.append(this_X_train_vec)
        print("prediction done")
        
        # get test vec rep
        print("predicting out sample")
        this_X_test = [get_X_train(str2sentence(ctt[2])) for ctt in rst_lst[(b_id+1)*n_samples : (b_id+2)*n_samples]]

        this_X_test_vec = np.zeros((len(this_X_test), 100))
        for a_id, atc in enumerate(this_X_test):
            this_atc_vec = np.zeros((1, 100))
            
            for sentence in atc:
                this_atc_vec += model.infer_vector(sentence[0])
            
            this_X_test_vec[a_id, :] = this_atc_vec / len(atc)
        
        X_test_vec_lst.append(this_X_test_vec)
        print("prediction done")
        
    return X_train_vec_lst, X_test_vec_lst


# In[ ]:


X_train_vec_lst, X_test_vec_lst = doc2vec_backtest(rst_lst)


# In[ ]:


with open("X_train_vec_lst", "wb") as fp:
    pickle.dump(X_train_vec_lst, fp)
    
with open("X_test_vec_lst", "wb") as fp:
    pickle.dump(X_test_vec_lst, fp)

