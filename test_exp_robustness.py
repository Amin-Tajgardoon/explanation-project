# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:49:28 2017

@author: mot16

tests how robust are explanations when changing the number of samples in LIME
uses LR_L1 model with smote resampling
"""
import numpy as np
import pandas as pd

from __future__ import print_function
import lime
import lime.lime_tabular

import io
from contextlib import redirect_stdout

from Model_selection import read_X_y
from Model_selection import get_estimator

from model_selection_3 import smote

def get_exp_instances(model, class_label, X_test, y_test):
    test_inst_list = list(np.where(y_test == class_label))[0]
    index_list = [x for x in list(np.where(model.predict(X_test) == class_label))[0] if x in test_inst_list]
    return index_list

def get_explanations(instances, model, X_train, X_test, output_path, 
                     num_samples_list, labels, model_regressor, seed_num):
    
    exp_df = pd.DataFrame(columns=['model', 'lime_num_samples', 'instance_num',
                                   'model_proba',
                                   'local_pred', 'local_R_2'])
    np.random.seed(seed_num)
    explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.as_matrix(), feature_names=list(X_train.columns), 
            categorical_features= list(range(X_train.shape[1])), 
            class_names=['Non-Dire', 'Dire'], feature_selection='lasso_path', 
            discretize_continuous=False, discretizer=None, verbose=True)
    
    for num_samples in num_samples_list:
        
        for i in instances:
            
            print('num_samples: ', num_samples)
            print('explanation for instance: ', i)
            ## to capture local model's R^2 from LIME's stdout
            str_io = io.StringIO() 
            with redirect_stdout(str_io):
     
                np.random.seed(seed_num)
                exp = explainer.explain_instance(X_test.iloc[i,:], 
                            model.predict_proba, 
                            num_features=10, labels=labels,
                            num_samples=num_samples,
                            distance_metric='euclidean',
                            model_regressor=model_regressor)
            stdout = str_io.getvalue()
            start = stdout.find('[ ') + 2
            end = stdout.find(']', start)
            
            local_pred = stdout[start:end]                               
            exp_dict = dict(exp.as_list(label=1))
            exp_dict['model'] = 'LR_L1'
            exp_dict['lime_num_samples'] = num_samples
            exp_dict['instance_num'] = i
            exp_dict['model_proba'] = exp.predict_proba[1]
            exp_dict['local_pred']= local_pred
            exp_dict['local_R_2']= exp.score
            exp_df = exp_df.append(exp_dict, ignore_index=True)
        exp_df.to_csv(output_path, index=False)

    return exp_df

def main():
    seed_num = 0
    np.random.seed(seed_num)
    
    X_train, y_train = read_X_y('../data/port_train.csv')
    X_test, y_test = read_X_y('../data/port_test.csv')
    
    X_res, y_res = smote(X_train, y_train, 200, 5, 100, seed_num)
    model = get_estimator('LR_L1', 1, seed_num)
    model.fit(X_res, y_res)
    tp_inst = get_exp_instances(model, 1, X_test, y_test)
    num_samples_list = [5000, 10000, 15000, 20000, 30000]
    #exp_df = get_explanations(tp_inst, model, X_res, X_test,
    #                          output_path="../output/LR_L1_TP_Explanations.csv",
    #                          num_samples_list=num_samples_list, labels=[1],
    #                          model_regressor=None, seed_num=seed_num)
    exp_df = get_explanations(tp_inst, model, X_train, X_test,
                              output_path="../output/LR_L1_TP_Explanations2.csv",
                              num_samples_list=num_samples_list, labels=[1],
                              model_regressor=None, seed_num=seed_num)
    
    ## get supportive and opposing counts    
    ncols = exp_df.shape[1]                         
    tp_sup_counts = exp_df.groupby('lime_num_samples').apply(
                    lambda g: g.iloc[:,6:ncols][g.iloc[:,6:ncols]>0].count(axis=0))
                  
    tp_opp_counts = exp_df.groupby('lime_num_samples').apply(
                    lambda g: g.iloc[:,6:ncols][g.iloc[:,6:ncols]<0].count(axis=0))
    
    ## get average weights and concat with counts   
    means = exp_df.groupby('lime_num_samples').apply(
                lambda g: g.iloc[:,6:ncols].mean(axis=0))
    summary = pd.concat({'supporting_counts':tp_sup_counts,
                           'opposing_counts':tp_opp_counts,
                           'avrg_weight':means})
    ## swap rows and columns of summary df
    summary_swapped = summary.swapaxes(0,1).swaplevel(0,1,axis=1).sort_index(axis=1, level=0)
    summary_swapped.to_csv('../output/LR_L1_exp_summary.csv')
    
    ## write 1 summary file per lime_num_samples
    for n_sample in summary_swapped.columns.levels[0]:
        df = summary_swapped[n_sample]
        df = df.assign(temp=np.abs(df['avrg_weight'])).sort_values(by='temp', ascending=False).drop('temp', axis=1)
        df = df.iloc[1:10,:].sort_values('supporting_counts', ascending=False)
        df.to_csv('../output/LR_L1_nsample_' + str(n_sample) + '_exp_summary.csv')

    
    
    