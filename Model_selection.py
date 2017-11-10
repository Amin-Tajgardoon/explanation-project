# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:39:03 2017

@author: mot16
"""

import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def read_X_y(file, set_0_1 = True, zero = 1, one = 2):
    """
    reads file and sets class values to 0 and 1 using the mapping values
    """
    df = pd.read_csv(file)
    X = df.iloc[:,0:df.shape[1]-1]
    y = df.iloc[:,df.shape[1]-1]
    if(set_0_1):
        y = set_y_0_1(y, zero = 1, one = 2)
    return X, y

def set_y_0_1(y, zero = 1, one = 2):
    y[y == zero] = 0
    y[y == one] = 1
    return y
    
def test_model(estimator, X_test, y_test):
    probs = estimator.predict_proba(X_test)
    return get_measures(probs[:,1], estimator.predict(X_test), y_test)
    
def resample(X, y, method, seed_num):
    """
    supporting resampling methods: SMOTE, SMOTEENN, SMOTETomek
    """
    if method == 'SMOTE':
        resample = SMOTE(kind='borderline1', ratio='minority', random_state=seed_num)
    elif method == 'SMOTEENN':
        resample = SMOTEENN(random_state=seed_num)
    elif method == 'SMOTETomek':
        resample = SMOTETomek(random_state=seed_num)
    else:
        raise Exception('Resampling method, %s, is not supported!' % method)
    
    X_resampled, y_resampled = resample.fit_sample(X, y)
    print(method + ': ' + 'X_resampled shape: ' + str(X_resampled.shape))
    
    return X_resampled, y_resampled
    
def cross_validate(estimator, skf, X, y, resample_method=None):
    auroc_list = []
    f1_list = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        if resample_method != None:
            X_train, y_train = resample(X_train, y_train, resample_method, seed_num)
        estimator.fit(X_train, y_train)
        auroc, f1, _, _ = test_model(estimator, X_test, y_test)
        auroc_list.append(auroc)
        f1_list.append(f1)
    return np.mean(auroc_list), np.mean(f1_list)

def get_estimator(est, par, seed_num):
    if est == 'RF':
        estimator = RandomForestClassifier(n_estimators=int(par), random_state=seed_num,
                                           class_weight=None)
    elif est == 'SVM':
        estimator = SVC(C=par, class_weight=None, kernel='rbf', probability=True,
                        random_state=seed_num)
                        
    elif est == 'NB':
        estimator = MultinomialNB(alpha=par, class_prior=None, fit_prior=True)
    
    elif est == 'LR_L1':
        estimator = LogisticRegression(penalty='l1', C=par, random_state=seed_num)
        
    elif est == 'LR_L2':
        estimator = LogisticRegression(penalty='l2', C=par, random_state=seed_num)
    else:
        raise Exception('Estimator "%s" is not supported!' % est)
    
    return estimator
    
def get_measures(probs, y_pred, y_true):
    auroc = roc_auc_score(y_true, probs)
    #f1 = f1_score(y_test, estimator.predict(X_test), average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, pos_label = 1, average='binary')
    return auroc, f1, precision, recall


if __name__ == '__main__':
    
    seed_num = 0
    np.random.seed(seed_num)
    min_auc = .84
    
    X_train, y_train = read_X_y('../data/port_train.csv')
    # X_smote, y_smote = read_X_y('../data/port_train_smote.csv')
    X_test, y_test = read_X_y('../data/port_test.csv')
    
    n_trees = np.linspace(1000, 10000, num=10, dtype='int')
    C = np.arange(0.1, 5, .5)
    nb_alphas = np.arange(0, 250, 50)
    estimators = [('RF', n_trees),('SVM', C),('NB', nb_alphas),('LR_L1', C),('LR_L2', C)]
    
    skf = StratifiedKFold(n_splits=10, random_state = seed_num)
    
    resample_methods = [None, 'SMOTE', 'SMOTEENN', 'SMOTETomek']
    cv_results = {'resample_method': [], 'estimator': [], 'param': [], 'auroc': [], 'f1': []}
    
    loop_counter = 0
    
    ## optimizes parameters for best F1
    for est, params in estimators:
        f1_max = -1
        best_param = -1
        best_resample = None
        print('Cur_estimator: ' + est)
        
        for resample_method in resample_methods:
    
            for par in params:
                loop_counter += 1
                print(est + ', loop_counter=' + str(loop_counter))
                
                cv_results['resample_method'].append(resample_method)
                cv_results['estimator'].append(est)
                cv_results['param'].append(par)
                
                estimator = get_estimator(est, par, seed_num)
                auroc, f1 = cross_validate(estimator, skf, X_train.as_matrix(), y_train.as_matrix(), resample_method)
    
                cv_results['auroc'].append(auroc)
                cv_results['f1'].append(f1)
            
                if auroc > min_auc:
                    if f1 > f1_max:
                        f1_max = f1
                        best_param = par
                        best_resample = resample_method
                        
        print('Estimator: ' + est)
        print('f1_max= ' + str(f1_max))
        print('best_resample= ' + str(best_resample))
        print('best_param_value= ' + str(best_param))
        
    
    results = pd.DataFrame(cv_results, columns=['estimator','resample_method', 'param','f1', 'auroc'])
    
    results.to_csv("../output/model_evaluation/model_selection_phase.csv", index=False)
    
    top_res = results.loc[results.auroc>=.83,:].groupby('estimator').apply(lambda g: g.sort_values(['f1','auroc'], ascending=False)).groupby('estimator').head(1)
    
    top_res.to_csv("../output/model_evaluation/model_selection_top_results_auc_greater_than_0.83.csv", index = False)
    
    
    ## test result of model selection on TEST set
    for est in top_res.estimator:
        ind = (top_res.estimator == est)
        model = get_estimator(est, top_res[ind]['param'].item(), seed_num)
        resample_method = top_res[ind]['resample_method'].item()
        if(resample_method == None):
            model.fit(X_train, y_train)
        else:
            X_res, y_res = resample(X_train, y_train, resample_method, seed_num)
            model.fit(X_res, y_res)
        print(est + ': auroc, f1, precision, recall:' + str(test_model(model, X_test, y_test)))
    

