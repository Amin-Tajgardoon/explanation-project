# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:53:41 2017

@author: mot16
"""
import io
from contextlib import redirect_stdout

from __future__ import print_function
import lime
import lime.lime_tabular

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from Model_selection import read_X_y, get_estimator, resample


def get_exp_instances(model_dict, class_label, X_test, y_test):
    inst_dict = {}
    test_inst_list = list(np.where(y_test == class_label))[0]
    for model_name, model in model_dict.items():
        index_list = [x for x in list(np.where(model.predict(X_test) == class_label))[0] if x in test_inst_list]
        inst_dict[model_name] = index_list
    return inst_dict

def get_common_instances(inst_dict):
    commons = set()
    for model in inst_dict:
        inst_set = set(inst_dict[model])
        if len(commons) == 0:
            commons = inst_set
        else:
            commons = commons.intersection(inst_set)
    return commons

def get_explanations(common_insts, model_dict, X_train_dict,
                     model_params_dict,
                     seed_num, X_train, X_test, output_path, 
                     num_samples,
                     labels, model_regressor):
    
    exp_df = pd.DataFrame(columns=['model', 'instance_num',
                                   'model_proba',
                                   'local_pred', 'local_R_2'])
    for m in model_dict:
        train_set = X_train_dict[str(model_params_dict.loc[
                                        model_params_dict.estimator == m,
                                        'resample_method'].item())]
        np.random.seed(seed_num)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            train_set, feature_names=list(X_train.columns), 
            categorical_features= list(range(X_train.shape[1])), 
            class_names=['Non-Dire', 'Dire'], feature_selection='lasso_path', 
            discretize_continuous=False, discretizer=None, verbose=True)
        
        for i in common_insts:
            
            print('explanation for instance: ', i)
            str_io = io.StringIO() 
            with redirect_stdout(str_io):
 
                np.random.seed(seed_num)
                exp = explainer.explain_instance(X_test.iloc[i,:], 
                            model_dict[m].predict_proba, 
                            num_features=10, labels=labels,
                            num_samples=num_samples,
                            distance_metric='euclidean',
                            model_regressor=model_regressor)
            stdout = str_io.getvalue()
            start = stdout.find('[ ') + 2
            end = stdout.find(']', start)
            
            local_pred = stdout[start:end]                               
            exp_dict = dict(exp.as_list(label=1))
            exp_dict['model'] = m
            exp_dict['instance_num'] = i
            exp_dict['model_proba'] = exp.predict_proba[1]
            exp_dict['local_pred']= local_pred
            exp_dict['local_R_2']= exp.score
            exp_df = exp_df.append(exp_dict, ignore_index=True)
        exp_df.to_csv(output_path, index=False)
    
    return exp_df
    

if __name__ == '__main__':
    seed_num = 0
    
    X_train, y_train = read_X_y('../data/port_train.csv')
    X_test, y_test = read_X_y('../data/port_test.csv')
    ## reads model selection results
    top_res = pd.read_csv("../output/model_evaluation/model_selection_top_results_auc_greater_than_0.83.csv")
    
    ## store datasets in dictionary
    X_train_dict = {}
    y_train_dict = {}
    for resample_method in np.unique(top_res['resample_method']):
        if(resample_method == None):
            X_train_dict['None'], y_train_dict['None'] = X_train, y_train
        
        else:
            X_train_dict[resample_method], y_train_dict[resample_method] = resample(
            X_train, y_train, resample_method, seed_num)
        
    ## set settings for all models (from model selection results)
    model_dict = {}
    for est in np.unique(top_res.estimator):
        ind = (top_res.estimator == est)
        model = get_estimator(est, top_res[ind]['param'].item(), seed_num)
        resample_method = str(top_res[ind]['resample_method'].item())
        X = X_train_dict[resample_method]
        y = y_train_dict[resample_method]
        model.fit(X, y)
        model_dict[est] = model
    
    ## explanations for tp predictions
    tp_inst_dict = get_exp_instances(model_dict, class_label=1, 
                                     X_test=X_test, y_test=y_test)
    
    common_tp_insts = get_common_instances(tp_inst_dict)
    
    tp_exp_df = get_explanations(common_tp_insts, model_dict,
                                 X_train_dict, top_res, seed_num, X_train, X_test,
                                 output_path="../output/tp_explanations.csv")
    
    ##explanations for tp predictions
    tn_inst_dict = get_exp_instances(model_dict, 0, X_test, y_test)
    
    common_tn_insts = get_common_instances(tn_inst_dict)
    
    tn_exp_df = get_explanations(common_tn_insts, model_dict,
                                 X_train_dict, top_res, seed_num, X_train, X_test,
                                 output_path="../output/tn_explanations.csv")

    