# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:06:08 2017

@author: mot16
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import pandas as pd

base_3 = pd.read_csv("../data/base3_imputed.csv")

g = sns.factorplot(x="211.CXRTYPE", col="217.DIREOUT",
                    data=base_3, kind="count",
                    size=4, aspect=.7)

g = sns.factorplot(x="24.PTRACE", col="217.DIREOUT",
                   data=base_3, kind="count",
                 size=4, aspect=.7)

g = sns.factorplot(x="204.SOXYGEN", col="217.DIREOUT",
                   data=base_3, kind="count",
                   size=4, aspect=.7)
