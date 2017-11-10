
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


# In[8]:

## PREPARE DATA

train = np.genfromtxt ('data/port_train.csv', delimiter=",", dtype=int, names=None, skip_header=1)
test = np.genfromtxt ('data/port_test.csv', delimiter=",", dtype=int, names=None, skip_header=1)
train_X = train[:,0:156]
train_Y = train[:,156]
test_X = test[:,0:156]
test_Y = test[:,156]
train_Y[train_Y == 1] = 0
train_Y[train_Y == 2] = 1
test_Y[test_Y == 1] = 0
test_Y[test_Y == 2] = 1

df = pd.read_csv('data/port_train_smote.csv')
train_smote = df.as_matrix()
train_smote_X = train_smote[:,0:156]
train_smote_Y = train_smote[:,156]


# In[ ]:




# In[3]:

## Prepare dataframe to save evaluation metrics
evals = pd.DataFrame(index=['AUROC','Precision','Recall','F1','TNR(Specificity)','TP','FP','TN','FN'],
                    columns=['RF','RF_SMOTE','NB','NB_SMOTE','SVM','SVM_SMOTE','LR','LR_SMOTE'])


# In[ ]:




# In[881]:

## Naive Bayes

np.random.seed(0)
nb = MultinomialNB(alpha=20, class_prior=None, fit_prior=True)
nb.fit(train_X, train_Y)
nb_probs = nb.predict_proba(test_X)
auroc = roc_auc_score(test_Y, nb_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, nb.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, nb.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[882]:

evals['NB'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[884]:

## NB - SMOTE

np.random.seed(0)
nb = MultinomialNB(alpha=200, class_prior=None, fit_prior=True)
nb.fit(train_smote_X, train_smote_Y)
nb_probs = nb.predict_proba(test_X)
auroc = roc_auc_score(test_Y, nb_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, nb.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, nb.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[885]:

evals['NB_SMOTE'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[887]:

## RANDOM FOREST

rf = RandomForestClassifier(n_estimators=1000, random_state=0, verbose=False, class_weight=None)
rf.fit(train_X, train_Y)
rf.probs = rf.predict_proba(test_X)
auroc = roc_auc_score(test_Y, rf_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, rf.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, rf.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[888]:

evals['RF'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[340]:

## RANDOM FOREST - SMOTE


# In[6]:

rf = RandomForestClassifier(n_estimators=1000, random_state=0, class_weight=None, verbose=False)
rf.fit(train_smote_X, train_smote_Y)
rf_probs = rf.predict_proba(test_X)
auroc = roc_auc_score(test_Y, rf_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, rf.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, rf.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[891]:

evals['RF_SMOTE'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[893]:

## SVM

svc = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=0, shrinking=True,
    tol=0.001, verbose=False)

svc.fit(train_X, train_Y)
svc_probs = svc.predict_proba(test_X)
auroc = roc_auc_score(test_Y, svc_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, svc.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, svc.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[894]:

evals['SVM'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[896]:

## SVM - SMOTE
svc = SVC(C=1.5, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=0, shrinking=True,
    tol=0.001, verbose=True)

svc.fit(train_smote_X, train_smote_Y)
svc_probs = svc.predict_proba(test_X)
auroc = roc_auc_score(test_Y, svc_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, svc.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, svc.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[897]:

evals['SVM_SMOTE'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[899]:

## Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', tol=0.0001, C=0.9, fit_intercept=True, 
                        intercept_scaling=1, class_weight='balanced', random_state=0, 
                        solver='liblinear', max_iter=100, verbose=0)
lr.fit(train_X, train_Y)
lr_probs = lr.predict_proba(test_X)
auroc = roc_auc_score(test_Y, lr_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, lr.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, lr.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[900]:

evals['LR'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[ ]:




# In[901]:

## Logistic Regression - SMOTE

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', tol=0.0001, C=.2, fit_intercept=True, 
                        intercept_scaling=1, class_weight=None, random_state=0, 
                        solver='liblinear', max_iter=100, verbose=0)
lr.fit(train_smote_X, train_smote_Y)
lr_probs = lr.predict_proba(test_X)
auroc = roc_auc_score(test_Y, lr_probs[:,1])
(prec, rec, f, _) = precision_recall_fscore_support(test_Y, lr.predict(test_X), average='binary')
(tn, fp, fn, tp) = confusion_matrix(test_Y, lr.predict(test_X)).ravel()
tnr = tn/(tn+fp)


# In[902]:

evals['LR_SMOTE'] = [auroc,prec, rec, f, tnr, tp, fp, tn, fn]


# In[903]:

evals


# In[905]:

## write evals

evals.to_csv("output/model_evaluation/evaluation_measures/evals.csv")


# In[ ]:




# In[9]:

## LIME

from __future__ import print_function
import lime
import lime.lime_tabular
np.random.seed(0)
explainer = lime.lime_tabular.LimeTabularExplainer(
    train_smote_X, training_labels=None, feature_names=list(df.columns[df.columns != '217.DIREOUT']), 
    categorical_features= list(range(0,train_X.shape[1]-1)), categorical_names=None, kernel_width=None, verbose=False,
    class_names=['Non-Dire', 'Dire'], feature_selection='highest_weights', discretize_continuous=False, discretizer='quartile')


# In[ ]:




# In[699]:

## TP indices

rf_tp_ind = [x for x in list(np.where(rf.predict(test_X) == 1))[0] if x in list(np.where(test_Y == 1)[0])] 
nb_tp_ind = [x for x in list(np.where(nb.predict(test_X) == 1))[0] if x in list(np.where(test_Y == 1)[0])] 
svc_tp_ind = [x for x in list(np.where(svc.predict(test_X) == 1))[0] if x in list(np.where(test_Y == 1)[0])]
lr_tp_ind = [x for x in list(np.where(lr.predict(test_X) == 1))[0] if x in list(np.where(test_Y == 1)[0])]

tp_inds = list(set(set(set(rf_tp_ind).intersection(set(nb_tp_ind))).intersection(svc_tp_ind)).intersection(lr_tp_ind))


# In[698]:

(len(rf_tp_ind), len(nb_tp_ind), len(svc_tp_ind), len(lr_tp_ind))


# In[689]:

## TN indices

y_0_ind = list(np.where(test_Y == 0))[0]

rf_tn_ind = [x for x in list(np.where(rf.predict(test_X) == 0))[0] if x in y_0_ind] 
nb_tn_ind = [x for x in list(np.where(nb.predict(test_X) == 0))[0] if x in y_0_ind] 
svc_tn_ind = [x for x in list(np.where(svc.predict(test_X) == 0))[0] if x in y_0_ind]
lr_tn_ind = [x for x in list(np.where(lr.predict(test_X) == 0))[0] if x in y_0_ind]

tn_inds = list(set(set(set(rf_tn_ind).intersection(set(nb_tn_ind))).intersection(svc_tn_ind)).intersection(lr_tn_ind))


# In[ ]:




# In[ ]:




# In[703]:

## RF - TP - Feature exp

d = {}
for i in range(len(rf_tp_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[rf_tp_ind[i]], rf.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        d[f].append(w)

rf_tp_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))
rf_tp_weights.mean().sort_values(ascending=False)[0:15]


# In[ ]:




# In[772]:

## NB - TP - Feature exp

d = {}
for i in range(len(nb_tp_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[nb_tp_ind[i]], nb.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        d[f].append(w)

nb_tp_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[773]:

## SVM - TP - Feature exp

d = {}
for i in range(len(svc_tp_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[svc_tp_ind[i]], svc.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        d[f].append(w)

svc_tp_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[774]:

## LR - TP - Feature exp

d = {}
for i in range(len(lr_tp_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[lr_tp_ind[i]], lr.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        d[f].append(w)

lr_tp_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[ ]:




# In[816]:

## TP mean values

rf_sorted_means = rf_tp_weights.mean().sort_values(ascending=False)
tp_means = {}
tp_means['rf_features'] = rf_sorted_means.index.values
tp_means['rf_feature_mean'] = list(rf_sorted_means)
tp_means['rf_feature_std'] = list(rf_tp_weights.std(ddof=1)[rf_sorted_means.index.values])
tp_means['rf_feature_count'] = list(rf_tp_weights[tp_means['rf_features']].notnull().sum(axis=0))

nb_sorted_means = nb_tp_weights.mean().sort_values(ascending=False)
tp_means['nb_features'] = nb_sorted_means.index.values
tp_means['nb_feature_mean'] = list(nb_sorted_means)
tp_means['nb_feature_std'] = list(nb_tp_weights.std(ddof=1)[nb_sorted_means.index.values])
tp_means['nb_feature_count'] = list(nb_tp_weights[tp_means['nb_features']].notnull().sum(axis=0))

svc_sorted_means = svc_tp_weights.mean().sort_values(ascending=False)
tp_means['svc_features'] = svc_sorted_means.index.values
tp_means['svc_feature_mean'] = list(svc_sorted_means)
tp_means['svc_feature_std'] = list(svc_tp_weights.std(ddof=1)[svc_sorted_means.index.values])
tp_means['svc_feature_count'] = list(svc_tp_weights[tp_means['svc_features']].notnull().sum(axis=0))

lr_sorted_means = lr_tp_weights.mean().sort_values(ascending=False)
tp_means['lr_features'] = lr_sorted_means.index.values
tp_means['lr_feature_mean'] = list(lr_sorted_means)
tp_means['lr_feature_std'] = list(lr_tp_weights.std(ddof=1)[lr_sorted_means.index.values])
tp_means['lr_feature_count'] = list(lr_tp_weights[tp_means['lr_features']].notnull().sum(axis=0))


# In[823]:

tp_means_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tp_means.items() ]),
            columns=['rf_features','rf_feature_mean','rf_feature_std','rf_feature_count',
                    'nb_features','nb_feature_mean','nb_feature_std','nb_feature_count',
                    'svc_features','svc_feature_mean','svc_feature_std','svc_feature_count',
                    'lr_features','lr_feature_mean','lr_feature_std','lr_feature_count'])


# In[826]:

## Write outputs
tp_means_df.to_csv("output/tp_feature_means.csv", index=False)


# In[843]:

pd.concat([pd.DataFrame(lr_tp_ind, columns=['test_index']),
           lr_tp_weights], axis = 1).to_csv("output/lr_tp_feature_weights.csv", index=False)


# In[ ]:




# In[844]:

## RF - TN - Feature exp

d = {}
for i in range(len(rf_tn_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[rf_tn_ind[i]], rf.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        w = -w
        d[f].append(w)

rf_tn_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[845]:

## NB - TN - Feature exp

d = {}
for i in range(len(nb_tn_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[nb_tn_ind[i]], nb.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        w = -w
        d[f].append(w)

nb_tn_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[846]:

## SVC - TN - Feature exp

d = {}
for i in range(len(svc_tn_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[svc_tn_ind[i]], svc.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        w = -w
        d[f].append(w)

svc_tn_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[847]:

## LR - TN - Feature exp

d = {}
for i in range(len(lr_tn_ind)):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[lr_tn_ind[i]], lr.predict_proba, num_features=test_X.shape[1], top_labels=2)
    exp_list = exp.as_list()
    for (f,w) in exp_list:
        if not f in d:
            d[f] = []
        w = -w
        d[f].append(w)

lr_tn_weights = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items() ]))


# In[ ]:




# In[849]:

## TN mean values

tn_means = {}

rf_tn_sorted_means = rf_tn_weights.mean().sort_values(ascending=False)

tn_means['rf_features'] = rf_tn_sorted_means.index.values
tn_means['rf_feature_mean'] = list(rf_tn_sorted_means)
tn_means['rf_feature_std'] = list(rf_tn_weights.std(ddof=1)[rf_tn_sorted_means.index.values])
tn_means['rf_feature_count'] = list(rf_tn_weights[tn_means['rf_features']].notnull().sum(axis=0))

nb_tn_sorted_means = nb_tn_weights.mean().sort_values(ascending=False)
tn_means['nb_features'] = nb_tn_sorted_means.index.values
tn_means['nb_feature_mean'] = list(nb_tn_sorted_means)
tn_means['nb_feature_std'] = list(nb_tn_weights.std(ddof=1)[nb_tn_sorted_means.index.values])
tn_means['nb_feature_count'] = list(nb_tn_weights[tn_means['nb_features']].notnull().sum(axis=0))

svc_tn_sorted_means = svc_tn_weights.mean().sort_values(ascending=False)
tn_means['svc_features'] = svc_tn_sorted_means.index.values
tn_means['svc_feature_mean'] = list(svc_tn_sorted_means)
tn_means['svc_feature_std'] = list(svc_tn_weights.std(ddof=1)[svc_tn_sorted_means.index.values])
tn_means['svc_feature_count'] = list(svc_tn_weights[tn_means['svc_features']].notnull().sum(axis=0))

lr_tn_sorted_means = lr_tn_weights.mean().sort_values(ascending=False)
tn_means['lr_features'] = lr_tn_sorted_means.index.values
tn_means['lr_feature_mean'] = list(lr_tn_sorted_means)
tn_means['lr_feature_std'] = list(lr_tn_weights.std(ddof=1)[lr_tn_sorted_means.index.values])
tn_means['lr_feature_count'] = list(lr_tn_weights[tn_means['lr_features']].notnull().sum(axis=0))


# In[ ]:




# In[850]:

tn_means_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tn_means.items() ]),
            columns=['rf_features','rf_feature_mean','rf_feature_std','rf_feature_count',
                    'nb_features','nb_feature_mean','nb_feature_std','nb_feature_count',
                    'svc_features','svc_feature_mean','svc_feature_std','svc_feature_count',
                    'lr_features','lr_feature_mean','lr_feature_std','lr_feature_count'])


# In[851]:

## Save TN Weight means

tn_means_df.to_csv("output/tn_feature_means.csv", index=False)


# In[855]:

## Save TN Weights

pd.concat([pd.DataFrame(svc_tn_ind, columns=['test_index']),
           svc_tn_weights], axis = 1).to_csv("output/svc_tn_feature_weights.csv", index=False)


# In[ ]:




# In[874]:

## Save sample explanations for TP & TN

for i in range(3):
    np.random.seed(0)
    exp = explainer.explain_instance(test_X[tn_inds[i]], lr.predict_proba, num_features=test_X.shape[1], top_labels=None)
    exp.save_to_file("output/lr_tn_explain_inst_" + str(tn_inds[i]) + ".html")


# In[ ]:




# In[856]:

np.savetxt('output/rf_probs.csv', rf_probs, delimiter=',', fmt='%.4f')


# In[860]:

np.savetxt('output/nb_probs.csv', nb_probs, delimiter=',', fmt='%.4f')


# In[861]:

np.savetxt('output/svc_probs.csv', svc_probs, delimiter=',', fmt='%.4f')


# In[862]:

np.savetxt('output/lr_probs.csv', lr_probs, delimiter=',', fmt='%.4f')


# In[ ]:




# In[271]:

y_score = rf.probs
y_test = test_Y

# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
print(__doc__)

import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




# In[ ]:



