from tqdm import tqdm
import pandas as pd
from itertools import product
from back_test import *
from utils import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn.linear_model import *
from naive_bayes_classifier.configure import *
from naive_bayes_classifier.load_data import str2word_bag
import numpy as np
import gc


def main():
    rate_se, rst_lst = load_data()

    rev_ar = np.vstack(get_reval(rst_lst, rate_se))

    model_lst = ["ExtraTreesClassifier(n_estimators=50)",
                 "BaggingClassifier(base_estimator=MultinomialNB())"
                 ]

    #y_lst = [get_y(rev_ar, lag, percentile / 100)[158:] for lag in [1, 2, 5, 10] for percentile in [10, 20, 30]]
    y_lst = [f"{percentile}, {lag}" for lag in [1, 2, 5, 10] for percentile in [10, 20, 30]]

    X_test_lst = np.load("d2v_X_test.npy")
    X_train_lst = np.load("d2v_X_train.npy")

    X_lst = ["X_test_lst"]

    look_back_lst = ["100", "200", "300"]

    rst_lst = []
    for idx, one_config in tqdm(enumerate(product(model_lst, X_lst, y_lst, look_back_lst))):
        #rst_lst.append((idx, rolling_test(*one_config)))
        #if idx == 20:
        #    print(one_config)
        rst_lst.append(one_config)

        with open("params_d2v_str.pkl", "wb") as fp:
            pickle.dump(rst_lst, fp)

        gc.collect()

    print(1)


if __name__ == "__main__":
    main()
