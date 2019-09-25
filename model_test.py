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


def main():
    rate_se, rst_lst = load_data()

    rev_ar = np.vstack(get_reval(rst_lst, rate_se))

    model_lst = [ExtraTreesClassifier(n_estimators=50),
                 BaggingClassifier(base_estimator=MultinomialNB())
                 ]

    X_dct = [str2word_bag(itm[2], STOP_CHARS, STOP_WORDS, to_lower=True) for itm in rst_lst]
    X_df = pd.DataFrame(X_dct).fillna(0)
    del X_dct

    X_title_dct = [str2word_bag(itm[0], STOP_CHARS, STOP_WORDS, to_lower=True) for itm in rst_lst]
    X_title_df = pd.DataFrame(X_title_dct).fillna(0)
    del X_title_dct

    X_total_df = pd.concat([X_title_df, X_df], axis=1)

    X_lst = [X_df.values, X_title_df, X_total_df.values]
    y_lst = [get_y(rev_ar, lag, quantile / 100) for lag in [1, 2, 5, 10] for quantile in [10, 20, 30]]

    look_back_lst = [100, 200, 300, 400, 500]

    rst_lst = []
    for idx, one_config in tqdm(enumerate(product(model_lst, X_lst, y_lst, look_back_lst))):
        rst_lst.append((idx, rolling_test(*one_config)))

    print(1)


if __name__ == "__main__":
    main()