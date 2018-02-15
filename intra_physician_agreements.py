# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:30:01 2018
@author: mot16

1- measures aggreeement between 3 physicians (pairwise kappa score)
2- Chi-2 test between agreement rates in majority real tp vs majority real tn  
3- Chi-2 test between disgreement rates in majority fake tp vs majority fake tn  
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from fleiss_kappa import fleiss_kappa
from scipy.stats import chi2_contingency


## read the pt_descriptions results collected from 3 physicians 
df = pd.read_csv("../output/pt_descriptions_results.csv")

## physician's evals as a list
sh = df.loc[df.physician == 'Shyam',:].iloc[:,-6:].values.flatten()
ml = df.loc[df.physician == 'Malar',:].iloc[:,-6:].values.flatten()
lc = df.loc[df.physician == 'Luca',:].iloc[:,-6:].values.flatten()

## 3 eval lists to a dataframe
evals = pd.DataFrame({'shyam':sh, 'malar':ml, 'luca':lc})
evals.replace("+",1,inplace=True)
evals.replace("-",0,inplace=True)

## overall agreement
print("fleiss_kappa, overall=",fleiss_kappa(evals, 3))

## agreement within real/fake explanations
fake_real = df.fake_real[df.physician == "Shyam"].apply(lambda x: np.array([x]*6))
fake_real = np.array(fake_real.tolist()).ravel()
evals['fake_real'] = fake_real
print("fleiss_kappa, within_real=",fleiss_kappa(evals.loc[evals.fake_real == 'real',:].iloc[:,0:3], 3))
print("fleiss_kappa, within_fake=",fleiss_kappa(evals.loc[evals.fake_real == 'fake',:].iloc[:,0:3], 3))

## agreement within real explanations
tp_tn = df.tp_tn[df.physician == "Shyam"].apply(lambda x: np.array([x]*6))
tp_tn = np.array(tp_tn.tolist()).ravel()
evals['tp_tn'] = tp_tn

e = evals[(evals.fake_real == 'real') & (evals.tp_tn == 'tp')].iloc[:,0:3]
print("fleiss_kappa, real_tp=",round(fleiss_kappa(e, 3),2))

e = evals.loc[(evals.fake_real == 'real') & (evals.tp_tn == 'tn'),:].iloc[:,0:3]
print("fleiss_kappa, real_tn=",round(fleiss_kappa(e, 3),2))

e = evals.loc[(evals.fake_real == 'fake') & (evals.tp_tn == 'tp'),:].iloc[:,0:3]
print("fleiss_kappa, fake_tp=",round(fleiss_kappa(e, 3),2))

e = evals.loc[(evals.fake_real == 'fake') & (evals.tp_tn == 'tn'),:].iloc[:,0:3]
print("fleiss_kappa, fake_tn=",round(fleiss_kappa(e, 3),2))



## pairwise kappa
print("kappa, overall, luca vs malar=",round(cohen_kappa_score(evals.iloc[:,0], evals.iloc[:,1]),2))
print("kappa, overall, luca vs shyam=",round(cohen_kappa_score(evals.iloc[:,0], evals.iloc[:,2]),2))
print("kappa, overall, shyam vs malar=",round(cohen_kappa_score(evals.iloc[:,1], evals.iloc[:,2]),2))

e = evals.loc[evals.fake_real == 'real',:]
print("kappa, real, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, real, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, real, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))

e = evals.loc[evals.fake_real == 'fake',:]
print("kappa, fake, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, fake, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, fake, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))

e = evals[(evals.fake_real == 'real') & (evals.tp_tn == 'tp')]
print("kappa, real_tp, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, real_tp, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, real_tp, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))

e = evals[(evals.fake_real == 'real') & (evals.tp_tn == 'tn')]
print("kappa, real_tn, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, real_tn, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, real_tn, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))

e = evals[(evals.fake_real == 'fake') & (evals.tp_tn == 'tp')]
print("kappa, fake_tp, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, fake_tp, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, fake_tp, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))

e = evals[(evals.fake_real == 'fake') & (evals.tp_tn == 'tn')]
print("kappa, fake_tn, luca vs malar=",round(cohen_kappa_score(e.luca, e.malar),2))
print("kappa, fake_tn, luca vs shyam=",round(cohen_kappa_score(e.luca, e.shyam),2))
print("kappa, fake_tn, shyam vs malar=",round(cohen_kappa_score(e.shyam, e.malar),2))


## chi-2 

## contingency tables
##          TP   TN
## disagree
## agree
disagree_real = [17, 22]
agree_real = [73, 68]
obs_real = np.array([disagree_real, agree_real])
chi2_contingency(obs_real)

disagree_fake = [22, 7]
agree_fake = [8, 23]
obs_fake = np.array([disagree_fake, agree_fake])
chi2_contingency(obs_fake)


## contingency tables
##          real   fake
## disagree
## agree
disagree = [31, 29]
agree = [141, 31]
obs = np.array([disagree, agree])
chi2_contingency(obs)


## extract features in which each physician disagreed with the rest
res = pd.read_csv("../output/all_explanation_features.csv")
evals['feature'] = res.feature
evals['pt_no'] = res.pt_no
evals['direction'] = res.weight.apply(lambda x: 'supportive' if x > 0 else 'contradictory')
diff_luca = evals[(evals.luca != evals.malar) & (evals.luca != evals.shyam)]
diff_luca = (diff_luca['tp_tn'] + ', '+ diff_luca['feature'] + ', ' + diff_luca['direction']).value_counts()
pd.DataFrame({'group_feature':diff_luca.index,'freq':diff_luca.values}, columns=['group_feature','freq']).to_csv("../output/diff_luca.csv", index=False)

diff_malar = evals[(evals.shyam != evals.malar) & (evals.malar != evals.luca)]
diff_malar = (diff_malar['tp_tn'] + ', '+ diff_malar['feature']+ ', ' + diff_malar['direction']).value_counts()
pd.DataFrame({'group_feature':diff_malar.index,'freq':diff_malar.values}, columns=['group_feature','freq']).to_csv("../output/diff_malar.csv", index=False)
