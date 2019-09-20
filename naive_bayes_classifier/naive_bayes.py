"""
Created by: Hanyuan Hu, netID: hh1924
"""

from sparse_matrix import SparseMatrix
from tqdm import tqdm
from math import log
from itertools import groupby


class MultinomialNB(object):

    def __init__(self, verbose=True, eps=1/50):
        """
        :param eps: if not observed in samples, what probability we would like to assume to see in the population
        """
        self._verbose = verbose
        self._eps = eps

    def fit(self, X, y):
        n = len(y)
        self._p_c = dict([(k, len(tuple(g)) / n) for k, g in groupby(sorted(y))])
        self._p_c_x = SparseMatrix({}, dim=3, zero_value=0)

        itr = tqdm(set(y)) if self._verbose else set(y)
        for c in itr:
            this_X = [X[i] for i in range(len(X)) if y[i] == c]

            tp = 1 / len(this_X)
            for row in this_X:
                for wd, wd_count in row.items():
                    self._p_c_x[c, wd, wd_count] += tp

    def predict(self, X):
        rst_lst = []

        itr = tqdm(X) if self._verbose else X
        for row in itr:
            max_log_like = None
            max_c = None

            for c in self._p_c:
                log_like = log(self._p_c[c])

                for wd, wd_count in row.items():
                    log_like += log(max(self._p_c_x[c, wd, wd_count], self._eps))

                if max_log_like is None or max_log_like < log_like:
                    max_log_like = log_like
                    max_c = c

            rst_lst.append(max_c)

        return rst_lst


class MultinomialNBWithAdjust(MultinomialNB):

    def __init__(self, verbose=True, eps=1/50, decay_factor=2):
        super().__init__(verbose, eps)
        self._decay_factor = decay_factor

    def predict(self, X):
        rst_lst = []

        itr = tqdm(X) if self._verbose else X
        for row in itr:
            max_log_like = None
            max_c = None

            for c in self._p_c:
                log_like = log(self._p_c[c])

                for wd, wd_count in row.items():
                    p = self._p_c_x[c, wd, wd_count]

                    if p == 0.0:
                        prev_count = 1
                        while wd_count - prev_count > 0 and p == 0:
                            p = self._p_c_x[c, wd, wd_count - prev_count] / self._decay_factor**prev_count
                            prev_count += 1

                    log_like += log(max(p, self._eps))

                if max_log_like is None or max_log_like < log_like:
                    max_log_like = log_like
                    max_c = c

            rst_lst.append(max_c)

        return rst_lst


if __name__ == "__main__":
    from load_data import load_data
    X_econ, y_econ, _ = load_data()
    model = MultinomialNB()
    model.fit(X_econ, y_econ)
    y_pred = model.predict(X_econ)
    print(1)