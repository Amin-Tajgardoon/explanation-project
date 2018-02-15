# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:44:47 2017
@author: mot16

Splits original data to train and test, removes irrelevant columns
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


base_3 = pd.read_csv("../data/base3_imputed.csv")
X = base_3.loc[:, base_3.columns != '217.DIREOUT']
y = base_3['217.DIREOUT']

## SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=1601,
                                                    test_size=686, 
                                                    random_state=42)

train = pd.concat([pd.DataFrame(X_train, columns=list(X.columns)), pd.DataFrame(y_train, columns=['217.DIREOUT'])], axis=1)

test = pd.concat([pd.DataFrame(X_test, columns=list(X.columns)), pd.DataFrame(y_test, columns=['217.DIREOUT'])], axis=1)

train.to_csv("../data/base3_imputed_train.csv", index=False)
test.to_csv("../data/base3_imputed_test.csv", index=False)


## Keep PORT columns + PATIENT IDS
port_train = pd.read_csv("../data/port_train.csv")
port_train_pid = train[np.insert(port_train.columns,0,'1.STNUM')]
port_test_pid = test[np.insert(port_train.columns,0,'1.STNUM')]

port_train_pid.to_csv("../data/port_train_pid.csv",  index=False)
port_test_pid.to_csv("../data/port_test_pid.csv",  index=False)

## REMOVE UNWANTED COLUMNS
to_remove = ['24.PTRACE','29.COBEFORY','31.SPUTBFOY', '33.BFPNSOBY', '35.BEFCPBRY','37.BEFFATGY','53.CNUMSYM'
,'57.PRIATBNM','63.CPNHOSPN','82.CNUMCOMO','106.NUMIMMUN','121.PTEDUC','204.SOXYGEN','205.CO2RETEN','211.CXRTYPE','155.ALERT','209.CXRPRCNT','195.O2SATABG','208.CXRCLOBE','194.FIO2POX','202.FIO2ABG','203.FIO2ABGA']

port_train_new = port_train_pid.drop(to_remove,axis=1)
port_test_new = port_test_pid.drop(to_remove,axis=1)
port_train_new.to_csv("../data/port_train_new.csv",index=False)
port_test_new.to_csv("../data/port_test_new.csv",index=False)


