"""
Created by: Hanyuan Hu, netID: hh1924
"""

from functools import reduce


class SparseMatrix(object):

    def __init__(self, data_dct, dim, zero_value=0):
        self._data_dct = data_dct
        self._zero_value = zero_value
        self._dim = dim

    def __getitem__(self, item):
        if len(item) != self._dim:
            raise KeyError("Dim mismatch")

        try:
            return reduce(lambda x, y: x[y], (self._data_dct, *item))
        except KeyError:
            return self._zero_value

    def __setitem__(self, k, v):
        if len(k) != self._dim:
            raise KeyError("Dim mismatch")

        tmp = self._data_dct

        for i in k[:-1]:
            try:
                tmp = tmp[i]
            except KeyError:
                tmp[i] = {}
                tmp = tmp[i]

        tmp[k[-1]] = v


if __name__ == "__main__":
    sp_mat = SparseMatrix({1:{11:1, 12:2}, 2:{21:1}}, 2, 0)
    print(1)