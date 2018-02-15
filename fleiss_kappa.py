# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:26:50 2018

Fleiss Kappa score for multiple annotators
source: Wikipedia

@author: mot16
"""
import pandas as pd

## fleiss kappa, kappa for multiple annotators
## fleiss kappa: https://en.wikipedia.org/wiki/Fleiss%27_kappa
## p_cat = sum(cat_votes)/(n_subjects*n_raters)
## p_subject = (1 / n_raters^2 - n_raters) * ( sum(cat_vote^2) - n_raters)
## p_hat = avrg(p_subjects)
## p_hat_e = sum(p_cat^2)
## fk = (p_hat - p_hat_e) / (1 - p_hat_e)
## n_raters = 3
def fleiss_kappa(evals, n_raters):
    ## evals shape =(n_features, n_raters)    
    if evals.shape[1] != n_raters:
        raise Exception("number of columns does not match n_raters")
    
    fk_m = pd.DataFrame({'cat_0':n_raters-evals.sum(axis=1), 'cat_1':evals.sum(axis=1)})
    
    p_cats = fk_m.sum(axis=0)/ (fk_m.shape[0]*n_raters)
    p_subjs = ((n_raters*(n_raters-1))**-1) * ((fk_m**2).sum(axis=1)-n_raters)
    p_hat = p_subjs.mean()
    p_hat_e = (p_cats**2).sum()
    
    fk = (p_hat - p_hat_e) / (1 - p_hat_e)
    return fk