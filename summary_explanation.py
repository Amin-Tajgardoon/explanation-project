# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 09:24:50 2017

@author: mot16
"""
import pandas as pd
import numpy as np

def save_summary_exps(exps, sup_counts, opp_counts, output_dir, output_prefix ):
    
    means = exps.groupby('model').apply(
                lambda g: g.iloc[:,1:g.shape[1]].mean(axis=0))
    summary = pd.concat({'supporting_counts':sup_counts,
                           'opposing_counts':opp_counts,
                           'avrg_weight':means})
                           
    summary = summary.swapaxes(0,1).swaplevel(0,1,axis=1).sort_index(axis=1, level=0)
    
    summary.to_csv(output_dir + output_prefix + '_exp_summary.csv')
    
    for model in summary.columns.levels[0]:
        df = summary[model]
        df = df.assign(temp=np.abs(df['avrg_weight'])).sort_values(by='temp', ascending=False).drop('temp', axis=1)
        df = df.iloc[1:10,:].sort_values('supporting_counts', ascending=False)
        df.to_csv(output_dir + output_prefix + '_exp_summary_'+ model + '.csv')


if __name__ == '__main__':
    tp_exp = pd.read_csv("../output/tp_explanations.csv")
    
    tp_exp = tp_exp.loc[:, tp_exp.columns != 'instance_num']
    
    
    tp_sup_counts = tp_exp.groupby('model').apply(
                    lambda g: g[g>0].iloc[:,1:g.shape[1]].count(axis=0))
                  
    tp_opp_counts = tp_exp.groupby('model').apply(
                    lambda g: g[g<0].iloc[:,1:g.shape[1]].count(axis=0))
    
    save_summary_exps(tp_exp, tp_sup_counts, tp_opp_counts, '../output/', 'tp')
    
    
    
    tn_exp = pd.read_csv("../output/tn_explanations.csv")
    
    tn_exp = tn_exp.loc[:, tn_exp.columns != 'instance_num']
    
    tn_sup_counts = tn_exp.groupby('model').apply(
                    lambda g: g[g<0].iloc[:,1:g.shape[1]].count(axis=0))
                  
    tn_opp_counts = tn_exp.groupby('model').apply(
                    lambda g: g[g>0].iloc[:,1:g.shape[1]].count(axis=0))
    
    save_summary_exps(tn_exp, tn_sup_counts, tn_opp_counts, '../output/', 'tn')
                


