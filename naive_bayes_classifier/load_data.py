"""
Created by: Hanyuan Hu, netID: hh1924
"""

import os
from itertools import groupby
from configure import STOP_CHARS, STOP_WORDS


def get_hist(lst, key=lambda x: x):
    return dict([(k, len(tuple(g))) for k, g in groupby(sorted(lst, key=key))])


def str2word_bag(orig_str, stop_chars, stop_words, to_lower=False):
    this_stop_words = " ".join(stop_words)

    orig_str = orig_str.replace("\n", " ")

    if to_lower:
        orig_str = orig_str.lower()

    for c in stop_chars:
        orig_str = orig_str.replace(c, "")
        this_stop_words = this_stop_words.replace(c, "")

    this_stop_word_lst = this_stop_words.split(" ")
    del this_stop_words

    word_lst = orig_str.split(" ")

    for wd in this_stop_word_lst:
        try:
            word_lst.remove(wd)
        except ValueError:
            pass

    tt = get_hist(word_lst)
    try:
        tt.pop("")
    except KeyError:
        pass
    finally:
        return tt


def str2word_bag2(orig_str: str, stop_chars, stop_words, to_lower):
    this_stop_words = " ".join(stop_words)

    orig_str = orig_str.replace("\n", " ")

    if to_lower:
        orig_str = orig_str.lower()

    for c in stop_chars:
        orig_str = orig_str.replace(c, "")
        this_stop_words = this_stop_words.replace(c, "")

    this_stop_word_lst = this_stop_words.split(" ")
    del this_stop_words



    word_lst = orig_str.split(" ")
    tl = len(word_lst)
    for i in range(tl - 1):
        word_lst.append(word_lst[i] + word_lst[i+1])

    for wd in this_stop_word_lst:
        try:
            word_lst.remove(wd)
        except ValueError:
            pass

    tt = get_hist(word_lst)
    try:
        tt.pop("")
    except KeyError:
        pass
    finally:
        return tt


def load_data(stop_chars=STOP_CHARS, stop_words=STOP_WORDS, to_lower=False, str2wbg=str2word_bag):
    econ_region_lst = os.listdir("./economist")

    econ_article_lst = []
    for region in econ_region_lst:
        file_lst = os.listdir(os.path.join("./economist", region))

        for file in file_lst:
            with open(os.path.join("./economist", region, file), "r") as fp:
                econ_article_lst.append((file, fp.read()))

    onion_article_lst = []
    for file in os.listdir("./onion"):
        with open(os.path.join("./onion", file), "r") as fp:
            onion_article_lst.append((file, fp.read()))

    y_econ = [article[0].split(".")[0] for article in econ_article_lst]
    X_econ = [str2wbg(article[1], stop_chars, stop_words, to_lower) for article in econ_article_lst]

    X_onion = [str2wbg(article[1], stop_chars, stop_words, to_lower) for article in onion_article_lst]

    return X_econ, y_econ, X_onion
