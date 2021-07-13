from hashlib import scrypt
from numpy.core.records import fromrecords
import pandas as pd
import numpy as np
import random
from scipy.sparse.sputils import matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse, io
import math
from tqdm import tqdm


MAX_DF = 0.9
MIN_DF = 0.1
ATTR_NUM = 5
TREE_NUM = 100
MAX_DEPTH = 20

#vectorize
def vectorize(data):
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True,
                         max_df=MAX_DF, min_df=MIN_DF)
    tv_matrix = tv.fit_transform(data['posts'])
    print(tv_matrix.shape)
    return tv_matrix

#helper functions
def plurality_value(label_value):
    cnt = sum(label_value)
    if cnt >= len(label_value) / 2:
        return (1, cnt == len(label_value))
    else:
        return (0, cnt == 0)


def entropy(q):
    if q == 0 or q == 1:
        return 0
    else:
        return -q*np.log2(q) - (1-q)*np.log2(1-q)

#methods
def IG(attr_value, label_value, th):
    I = 0
    H = 0
    label_0 = label_value.count(0)
    I = entropy(label_0 / len(label_value))
    gt = [l for a, l in zip(attr_value, label_value) if a > th]
    gt_0 = gt.count(0)
    leq = [l for a, l in zip(attr_value, label_value) if a <= th]
    leq_0 = leq.count(0)
    if len(gt) != 0:
        H += len(gt)/len(attr_value)*entropy(gt_0/len(gt))
    if len(leq) != 0:
        H += len(leq)/len(attr_value)*entropy(leq_0/len(leq))
    return I - H


def GR(attr_value, label_value, th):
    I = 0
    H = 0
    IV = 0
    label_0 = label_value.count(0)
    I = entropy(label_0 / len(label_value))
    gt = [l for a, l in zip(attr_value, label_value) if a > th]
    gt_0 = gt.count(0)
    leq = [l for a, l in zip(attr_value, label_value) if a <= th]
    leq_0 = leq.count(0)
    if len(gt) != 0:
        H += len(gt)/len(attr_value)*entropy(gt_0/len(gt))
        IV -= len(gt) / len(attr_value) * np.log2(len(gt) / len(attr_value))
    if len(leq) != 0:
        H += len(leq)/len(attr_value)*entropy(leq_0/len(leq))
        IV -= len(leq) / len(attr_value) * np.log2(len(leq) / len(attr_value))
    IG_value = I - H
    return IG_value / (1 + IV)  # smooth


def Gini(attr_value, label_value, th):
    label_0 = label_value.count(0)
    gt = [l for a, l in zip(attr_value, label_value) if a > th]
    gt_0 = gt.count(0)
    leq = [l for a, l in zip(attr_value, label_value) if a <= th]
    leq_0 = leq.count(0)

    Gini_index = 0
    if len(leq) != 0:
        p_leq = leq_0/len(leq)
        Gini_leq = 1 - math.pow(p_leq, 2) - math.pow(1 - p_leq, 2)
        Gini_index += len(leq) / len(attr_value) * Gini_leq
    if len(gt) != 0:
        p_gt = gt_0/len(gt)
        Gini_gt = 1 - math.pow(p_gt, 2) - math.pow(1 - p_gt, 2)
        Gini_index += len(gt) / len(attr_value) * Gini_gt
    return -Gini_index  # then higher is better

#random forest building
def random_forest(data, tv_matrix, type, method=IG):
    forest = []
    for i in tqdm(range(TREE_NUM), desc='Building Random Forest'):
        tree = decision_tree(data, tv_matrix, type, method)
        if tree[0]:
            forest.append(tree)
    return forest


def decision_tree(data, tv_matrix, type, method):
    data_idx = [random.randint(0, tv_matrix.shape[0] - 1)
                for i in range(tv_matrix.shape[0])]
    tree = {}
    build_tree(data, tv_matrix, data_idx, data_idx, tree, type, method, 0)
    return (tree, data_idx)


def build_tree(data, tv_matrix, data_idx, last_data_idx, tree, type, method, depth):
    if len(data_idx) == 0:
        last_label_value = [data[type][i] for i in last_data_idx]
        return plurality_value(last_label_value)[0]

    label_value = [data[type][i] for i in data_idx]
    value, same = plurality_value(label_value)
    if depth == MAX_DEPTH or same:
        return value

    best_th = -1
    best_attr = -1
    best_score = -1
    attr_idx = random.sample(range(tv_matrix.shape[1]), ATTR_NUM)
    for attr in attr_idx:
        attr_value = [tv_matrix[i, attr] for i in data_idx]
        sorted_attr_value = attr_value.copy()
        sorted_attr_value.sort()
        for th_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            th = sorted_attr_value[math.floor(
                th_rate * len(sorted_attr_value))]
            score = method(attr_value, label_value, th)
            if th_rate == 0.1 or score > best_score:
                best_th = th
                best_attr = attr
                best_score = score
    tree['attr'] = best_attr
    tree['th'] = best_th
    data_leq_idx = [i for i in data_idx if tv_matrix[i, best_attr] <= best_th]
    data_gt_idx = [i for i in data_idx if tv_matrix[i, best_attr] > best_th]
    tree['leq'] = build_tree(
        data, tv_matrix, data_leq_idx, data_idx, {}, type, method, depth+1)
    tree['gt'] = build_tree(data, tv_matrix, data_gt_idx,
                            data_idx, {}, type, method, depth+1)
    return tree

#testing
def tree_test(tree, x):
    if x[0, tree['attr']] <= tree['th']:
        if tree['leq'] == 0 or tree['leq'] == 1:
            return tree['leq']
        else:
            return tree_test(tree['leq'], x)
    else:
        if tree['gt'] == 0 or tree['gt'] == 1:
            return tree['gt']
        else:
            return tree_test(tree['gt'], x)


def forest_test(forest, x):
    result = 0
    for tree in forest:
        result += tree_test(tree[0], x)
    if result >= len(forest) / 2:
        return 1
    else:
        return 0


def type_test(data, tv_matrix, type, method=IG):#without  over sampling
    print('Type=', type)
    forest = random_forest(data, tv_matrix, type, method)
    print('Margin=', margin(forest, data, tv_matrix, type))
    print('OOB error=', OOB(forest, data, tv_matrix, type))
    print('F1=', F1(forest, data, tv_matrix, type))
    print('\n')


def over_sampling_test(data, type, method=IG):
    sample_data = data.copy()
    sample_num = list(data[type]).count(0) - list(data[type]).count(1)
    print(sample_num)
    if sample_num > 0:  # 0 is more
        sample_list = [i for i in range(len(data)) if data[type][i] == 1]
        for i in tqdm(range(sample_num), desc='Over sampling', leave=False):
            rand_idx = random.randint(0, len(sample_list) - 1)
            sample_data = sample_data.append(data.iloc[sample_list[rand_idx]])
    else:
        sample_list = [i for i in range(len(data)) if data[type][i] == 0]
        for i in tqdm(range(-sample_num), desc='Over sampling', leave=False):
            rand_idx = random.randint(0, len(sample_list) - 1)
            sample_data = sample_data.append(data.iloc[sample_list[rand_idx]])
    sample_data = sample_data.sample(frac=1).reset_index(drop=True)
    tv_matrix = vectorize(sample_data)
    type_test(sample_data, tv_matrix, type, method)

#evaluation
def margin(forest, data, tv_matrix, type):
    total_margin = 0
    for i in tqdm(range(len(data)), desc='Calculating Margin', leave=False):
        cnt = 0
        for tree_idx in range(len(forest)):
            if tree_test(forest[tree_idx][0], tv_matrix[i]) == data[type][i]:
                cnt += 1
        total_margin += 2 * cnt/len(forest) - 1
    return total_margin/len(data)


def OOB(forest, data, tv_matrix, type):
    error_cnt = 0
    for data_idx in tqdm(range(len(data)), desc='Calculating OOB error', leave=False):
        tree_cnt = 0
        temp_forest = [f for f in forest if data_idx not in f[1]]
        error_cnt += (forest_test(temp_forest,
                                  tv_matrix[data_idx]) != data[type][data_idx])
    return error_cnt / len(data)


def F1(forest, data, tv_matrix, type):
    result = []
    for data_idx in tqdm(range(len(data)), desc='Calculating F1', leave=False):
        result.append(forest_test(forest, tv_matrix[data_idx]))
    TP = len([1 for y, l in zip(result, data[type]) if y == 1 and l == 1])
    FP = len([1 for y, l in zip(result, data[type]) if y == 1 and l == 0])
    TN = len([1 for y, l in zip(result, data[type]) if y == 0 and l == 1])
    FN = len([1 for y, l in zip(result, data[type]) if y == 0 and l == 0])

    P = TP/(TP + FP)
    R = TP/(TP + FN)
    F1_value = 2 * (P * R) / (P + R)
    return F1_value

if __name__ == '__main__':
    data = pd.read_csv('processed.csv', index_col=0)
    method = IG
    print('ATTR_NUM=', ATTR_NUM, ' TREE_NUM=', TREE_NUM,
          ' MAX_DEPTH=', MAX_DEPTH, ' method=', method, '\n')
    over_sampling_test(data, 'EI', method)
    over_sampling_test(data, 'SN', method)
    over_sampling_test(data, 'TF', method)
    over_sampling_test(data, 'JP', method)
