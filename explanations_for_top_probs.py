# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:29:49 2017
@author: mot16

generates LIME explanations for predictions in which the predictive model has high confidence
"""
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular

from Model_selection import read_X_y, get_estimator, resample

def true_ix(probs, y_test, clazz, seed_num):
    prob_ix = np.where((probs > .5) & (y_test.as_matrix() == clazz))
    prob_ix_sorted = probs[prob_ix].argsort()[::-1]
    return np.arange(0,probs.shape[0])[prob_ix][prob_ix_sorted]
    
    
def has_missing_in_base(exp_list, desc_vars, base, pid):
    exp_varnames = [f.split('=')[0] for f,_ in exp_list]
    base_varnames = desc_vars['base'][desc_vars.port.isin(exp_varnames)]
    base_values = base.loc[pid, base_varnames]
    has_missing = (base_values == -9).any()
    return has_missing
    
def explanations(true_ix, num_true, num_fake, X_train, X_test,
                      model, num_features, clazz, seed_num):
    
    base = pd.read_csv('../data/base.csv', index_col=0)
    desc_vars = pd.read_csv("../data/desc_vars_2.csv")    
    
    np.random.seed(seed_num)
    explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.as_matrix(), feature_names=list(X_train.columns), 
                categorical_features= list(range(X_train.shape[1])), 
                class_names=['Non-Dire', 'Dire'], feature_selection='lasso_path', 
                discretize_continuous=False, discretizer=None, verbose=True)
                
                
    real_exp_df = pd.DataFrame(columns=['model',
     'lime_num_samples','pid', 'model_proba','local_R_2'])
    
    fake_ix = np.array([], dtype=np.int)
    real_exp_size = 0          
    while (real_exp_size < num_true and true_ix.shape[0] > 0):
        
        rand_ix = np.random.choice(true_ix)
        true_ix = np.delete(true_ix, np.where(true_ix == rand_ix))
        
        np.random.seed(seed_num)
        exp = explainer.explain_instance(X_test.iloc[rand_ix,:], 
                    model.predict_proba, 
                    num_features=num_features, labels=(0,1),
                    num_samples=5000,
                    distance_metric='euclidean')
                    
        exp_list = exp.as_list(label=clazz)
        pid = X_test.index[rand_ix]
        if has_missing_in_base(exp_list, desc_vars, base, pid):
            fake_ix = np.append(fake_ix, rand_ix)
            continue
                
        exp_dict = dict(exp_list)
        exp_dict['model'] = 'LR_L1'
        exp_dict['lime_num_samples'] = 5000
        exp_dict['pid'] = pid
        exp_dict['model_proba'] = exp.predict_proba[clazz]
        exp_dict['local_R_2']= exp.score
        #exp_dict['missing_base_values'] = 'y' if has_missing else 'n'
        real_exp_df = real_exp_df.append(exp_dict, ignore_index=True)
        real_exp_size += 1
        print('real_exp_size=', real_exp_size)
    
    fake_ix = np.append(fake_ix, true_ix)
    
    fake_exp_df = pd.DataFrame(columns=['model',
     'lime_num_samples','pid', 'model_proba','local_R_2'])
     
    fake_exp_size = 0
    while (fake_exp_size < num_fake and fake_ix.shape[0] > 0):
        
        rand_ix = np.random.choice(fake_ix)
        fake_ix = np.delete(fake_ix, np.where(fake_ix == rand_ix))        
        
        np.random.seed(seed_num)
        exp = explainer.explain_instance(X_test.iloc[rand_ix,:], 
                    model.predict_proba, 
                    num_features=X_test.shape[1], labels=(0,1),
                    num_samples=5000,
                    distance_metric='euclidean')
        
        exp_list = exp.as_list(label=clazz)
        pid = X_test.index[rand_ix]
        if has_missing_in_base(exp_list[-num_features:], desc_vars, base, pid):
            continue
        
        exp_dict = dict(exp_list)
        exp_dict['model'] = 'LR_L1'
        exp_dict['lime_num_samples'] = 5000
        exp_dict['pid'] = pid
        exp_dict['model_proba'] = exp.predict_proba[clazz]
        exp_dict['local_R_2']= exp.score
        #exp_dict['missing_base_values'] = 'y' if has_missing else 'n'
        fake_exp_df = fake_exp_df.append(exp_dict, ignore_index=True)
        fake_exp_size += 1
        print('fake_exp_size=', fake_exp_size)
        
    return real_exp_df, fake_exp_df



seed_num = 0
np.random.seed(seed_num)

X_train, y_train = read_X_y('../data/port_train_new_subset.csv', set_0_1=False, ptid_as_idx=True)

X_test, y_test = read_X_y('../data/port_test_new_subset.csv', set_0_1=False, ptid_as_idx=True)

model = get_estimator('LR_L1',0.1,seed_num)
X_res, y_res = resample(X_train, y_train, 'SMOTE', seed_num)
model.fit(X_res, y_res)
probs = model.predict_proba(X_test)

tp_ix = true_ix(probs[:,1], y_test, 1, seed_num)

tp_ix = np.array([499])
exp_df, fake_exp_df = explanations(tp_ix, 30, 10, X_train, X_test,
                      model, 6, 1, seed_num)

#exp_df.to_csv("../output/true_tp_explanations.csv", index=False)
#fake_exp_df.to_csv("../output/fake_tp_explanations.csv", index=False)
#
#
#
#tn_ix = true_ix(probs[:,0],y_test,0,seed_num)
#tn_ix = tn_ix[probs[tn_ix,0] >= .8]
#
#exp_df, fake_exp_df = explanations(tn_ix, 30, 10, X_train, X_test,
#                      model, 6, 0, seed_num)           
#
#exp_df.to_csv( "../output/true_tn_explanations.csv", index=False)
#fake_exp_df.to_csv("../output/fake_tn_explanations.csv", index=False)
    