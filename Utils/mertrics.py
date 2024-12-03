import math
import numpy as np

from collections import Counter


def average_absolute_error(truth, predict):
    gt, et = np.array(truth), np.array(predict)
    return np.abs(et - gt).mean()


def average_relative_error(truth, predict):
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) / gt).mean()


def average_weighted_error(truth, predict):
    gt, et = np.array(truth), np.array(predict)
    return (np.abs(et - gt) * gt).mean()


def weighted_mean_relative_difference(truth, predict):
    wmrd1 = wmrd2 = 0
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))
    union_count = set(gt_count.keys()).union(set(et_count.keys()))

    for n in union_count:

        try:
            n1 = gt_count[n]
        except:
            n1 = 0 

        try:
            n2 = et_count[n]
        except:
            n2 = 0 
        
        wmrd1 += abs(n1 - n2)
        wmrd2 += ((n1 + n2) / 2)

    return wmrd1 / wmrd2


def entropy_absolute_error(truth, predict):
    n_key = len(truth)
    gt_epy = et_epy = 0
    gt_count = dict(Counter(truth))
    et_count = dict(Counter(predict))

    for i, c in gt_count.items():
        gt_epy += (i * (c / n_key) * math.log2(n_key / c))

    for i, c in et_count.items():
        et_epy += (i * (c / n_key) * math.log2(n_key / c))

    return abs(et_epy - gt_epy)