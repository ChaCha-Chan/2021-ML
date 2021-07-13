import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import svm

MAX_DF = 1.0
MIN_DF = 0.0
N_SPLITS = 5
KERNEL = 'linear'
SVM_C = 2


def SVM(single_type, train_index, test_index, data):
    X_train, X_test = data['posts'].iloc[train_index], data['posts'].iloc[test_index]
    y_train, y_test = data[single_type].iloc[train_index], data[single_type].iloc[test_index]
   
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, max_df=MAX_DF, min_df=MIN_DF)
    tv_train = tv.fit_transform(X_train)

    model = svm.SVC(C=SVM_C, kernel=KERNEL,
                    decision_function_shape='ovo')
    model.fit(tv_train, y_train)

    train_score = model.score(tv_train, y_train)
    print(single_type, 'train: ', train_score)

    tv_test = tv.transform(X_test)
    test_score = model.score(tv_test, y_test)
    print(single_type, 'test:  ', test_score)

    return test_score



if __name__ == '__main__':
    print(KERNEL, MAX_DF, MIN_DF, SVM_C)
    data = pd.read_csv('processed.csv')
    kf = KFold(n_splits=N_SPLITS)
    total_EI_score = 0
    total_SN_score = 0
    total_TF_score = 0
    total_JP_score = 0
    for train_index, test_index in kf.split(data):
        total_EI_score += SVM('EI', train_index, test_index, data)
        total_SN_score += SVM('SN', train_index, test_index, data)
        total_TF_score += SVM('TF', train_index, test_index, data)
        total_JP_score += SVM('JP', train_index, test_index, data)
        print('\n\n')
    print('avg_score:', 'EI:', total_EI_score/N_SPLITS, 'SN:', total_SN_score/N_SPLITS, 'TF:',total_TF_score/N_SPLITS, 'JP:',total_JP_score/N_SPLITS)

