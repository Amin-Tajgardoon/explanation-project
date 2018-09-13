# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:39:03 2017
@author: mot16

performs model selection by cross validation and resampling, saves the results into files
"""

import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def read_X_y(file, set_0_1 = True, zero = 1, one = 2, ptid_as_idx=False):
    """
    reads file and sets class values to 0 and 1 using the mapping values
    """
    df = pd.read_csv(file)
    X = df.iloc[:,0:df.shape[1]-1]
    y = df.iloc[:,df.shape[1]-1]
    if(set_0_1):
        y = set_y_0_1(y, zero = 1, one = 2)
    if(ptid_as_idx==True):
        ## SET Patient Ids as index and drop the PTID column
        X.index = X['1.STNUM']
        X.drop('1.STNUM',axis=1,inplace=True)
        y.index = X.index
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
    precision_list = []
    recall_list = []
    spec_list=[]
    probs = np.empty(y.shape)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        if resample_method != None:
            X_train, y_train = resample(X_train, y_train, resample_method, seed_num)
        estimator.fit(X_train, y_train)
        #test_model(estimator, X_test, y_test)
        p = estimator.predict_proba(X_test)
        probs[test_index] = p[:,1]
        auroc, f1, precision, recall, specifcity = get_measures(p[:,1], estimator.predict(X_test), y_test)
        auroc_list.append(auroc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        spec_list.append(specifcity)
    return probs, np.mean(auroc_list), np.mean(f1_list), np.mean(precision_list), np.mean(recall_list), np.mean(spec_list)

def get_estimator(est, par, seed_num, class_weight=None):
    if est == 'RF':
        estimator = RandomForestClassifier(n_estimators=int(par), random_state=seed_num,
                                           class_weight=class_weight)
    elif est == 'SVM':
        estimator = SVC(C=par, class_weight=class_weight, kernel='rbf', probability=True,
                        random_state=seed_num)
                        
    elif est == 'NB':
        estimator = MultinomialNB(alpha=par, class_prior=None, fit_prior=True)
    
    elif est == 'LR_L1':
        estimator = LogisticRegression(penalty='l1', C=par, random_state=seed_num, class_weight=class_weight)
        
    elif est == 'LR_L2':
        estimator = LogisticRegression(penalty='l2', C=par, random_state=seed_num, class_weight=class_weight)
    else:
        raise Exception('Estimator "%s" is not supported!' % est)
    
    return estimator
    
def get_measures(probs, y_pred, y_true):
    auroc = roc_auc_score(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, pos_label = 1, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn/(tn+fp)
    return auroc, f1, precision, recall, specificity


if __name__ == '__main__':
    
    seed_num = 0
    np.random.seed(seed_num)
#    min_auc = .84
    
    X_train, y_train = read_X_y('../data/port_train_new_subset.csv', set_0_1=False, ptid_as_idx=True)
    X_test, y_test = read_X_y('../data/port_test_new_subset.csv', set_0_1=False, ptid_as_idx=True)
    
    n_trees = [500]#, 1000, 3000]#np.linspace(1000, 10000, num=5, dtype='int')
    C = np.array([0.1, 1, 10])
    nb_alphas = [10]#[0, 0.1, 1, 10, 100] #np.arange(0, 250, 50)
    estimators = [('RF', n_trees),('SVM', C),('NB', nb_alphas),('LR_L1', C),('LR_L2', C)]
     #estimators = [('LR_L1', C)]     
    
    skf = StratifiedKFold(n_splits=10, random_state = seed_num)
    
    resample_methods = ['SMOTE']#[None, 'SMOTE']
    cv_results = {'resample_method': [], 'estimator': [], 'param': [], 'auroc': [], 'f1': [], 'precision': [], 'recall': [], 'specificity':[]}
    
    loop_counter = 0
    
    ## optimizes parameters for best F1
    for est, params in estimators:
        print('Cur_estimator: ' + est)
        
        for resample_method in resample_methods:
    
            for par in params:
                loop_counter += 1
                print(est + ', loop_counter=' + str(loop_counter))
                
                cv_results['resample_method'].append(resample_method)
                cv_results['estimator'].append(est)
                cv_results['param'].append(par)
                
                estimator = get_estimator(est, par, seed_num)
                probs, auroc, f1, prec, recall, specificity = cross_validate(estimator, skf, X_train.as_matrix(), y_train.as_matrix(), resample_method)
                np.savetxt("../output/probs_" + est + "_" + str(par) + "_" + resample_method + ".csv", probs, delimiter=",")
                cv_results['auroc'].append(auroc)
                cv_results['f1'].append(f1)
                cv_results['precision'].append(prec)
                cv_results['recall'].append(recall)
                cv_results['specificity'].append(specificity)
                
                
        
    
    results = pd.DataFrame(cv_results, columns=['estimator','resample_method', 'param','f1', 'auroc','precision', 'recall','specificity'])
    
    results.to_csv("../output/model_evaluation/model_selection_new_port_7.csv", index=False)
    
    top_res = results.groupby('estimator').apply(lambda g: g.sort_values(['f1','auroc'], ascending=False)).groupby('estimator').head(1)
    
    top_res.round(2).to_csv("../output/model_evaluation/model_selection_top_results.csv", index = False)
    
    
    ## test result of model selection on TEST set
    test_result = pd.DataFrame(columns=['estimator', 'resample_method', 'param', 'auroc', 'f1','precision',
       'recall', 'specificity'])
    for est in top_res.estimator:
        idx = (top_res.estimator == est)
        param = top_res[idx]['param'].item()
        model = get_estimator(est, param, seed_num)
        resample_method = top_res[idx]['resample_method'].item()
        if(resample_method == None):
            model.fit(X_train, y_train)
        else:
            X_res, y_res = resample(X_train, y_train, resample_method, seed_num)
            model.fit(X_res, y_res)
        row = [est, resample_method, param] + list(test_model(model, X_test, y_test))
        test_result = test_result.append(pd.DataFrame(np.array([row]),columns=test_result.columns.tolist()))
        #print(est + ': auroc, f1, precision, recall, specificity:' + str(test_model(model, X_test, y_test)))
    
    test_result.round(2).to_csv("../output/model_evaluation/model_selection_test_results_for_top_models.csv", index = False)
    
