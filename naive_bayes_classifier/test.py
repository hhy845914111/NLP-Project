import pickle
from load_data import *

with open("../rst.pkl", "rb") as fp:
    rst_lst = pickle.load(fp)

X = [str2word_bag(itm[2], STOP_CHARS, STOP_WORDS, to_lower=True) for itm in rst_lst]

print(1)