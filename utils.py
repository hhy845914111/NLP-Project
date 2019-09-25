import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score


def load_data(y_file = "./USTREASURY-REALYIELD.csv"):
    with open("rst.pkl", "rb") as fp:
        rst_lst = pickle.load(fp)

    rate_se = pd.read_csv(y_file, index_col=0)["10 YR"]
    rate_se.index = pd.DatetimeIndex(rate_se.index)
    rate_se = rate_se.sort_index()

    return rate_se, rst_lst


def evaluate(y_true, y_pred):
    print(f"""acuracy: {accuracy_score(y_true, y_pred)}
    precision: {precision_score(y_true, y_pred, average="micro")}
    recall: {recall_score(y_true, y_pred, average="micro")}
    f1: {f1_score(y_true, y_pred, average="micro")}""")


def get_reval(rst_lst, rate_se, prev_n=10, rev_n=10):
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


def get_y(rev_ar, lag, p):
    y_raw = rev_ar[:, 9 + lag]
    y_neg = (y_raw < np.quantile(y_raw, p)).astype(np.int)
    y_pos = (y_raw > np.quantile(y_raw, 1 - p)).astype(np.int)
    return np.zeros_like(y_raw) - y_neg + y_pos


def get_alpha_decay(reval_ar, y):
    return np.mean(reval_ar * (y.reshape(-1, 1)), axis=0)

