# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:22:23 2017
@author: mot16

Function to generate customized plot for LIMe explanations
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def draw_exp_plot(exp_df, class_one_prob, out_path):
    plt.figure(figsize=(10,3))
    
    sns.set_style('ticks')
    gs = gridspec.GridSpec(4, 7)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[:, 4:])
    
    probs_df = pd.DataFrame({'outcome':['yes', 'no'], 'prob':[class_one_prob,1-class_one_prob]})
    prob_ax =sns.barplot(y="outcome", x="prob", data=probs_df, color='blue', ax=ax1)
    for i in range(probs_df.shape[0]):
        p=probs_df['prob'][i]
        prob_ax.text(p+0.02, i, str(p),  ha='left', va= 'center', size='15')
        
    prob_ax.set(xlabel='', ylabel='')
    prob_ax.axes.set_title('Dire outcome probability', fontsize='15',weight='bold')
    prob_ax.yaxis.set_tick_params(labelsize=15)
    prob_ax.yaxis.set_tick_params(length=0)
    sns.despine(ax=prob_ax, bottom=True)
    prob_ax.axes.get_xaxis().set_visible(False)
    
    
    exp_ax=sns.barplot(x="weights", y="labels", data=exp_df, palette=['green' if w > 0 else 'red' for w in exp_df.weights.values], ax=ax2)
    exp_ax.set(xlabel='', ylabel='')
    exp_ax.axes.text(0.01, -.58, 'Supportive', ha='left', fontsize=15, color = 'green', fontweight='bold' )
    exp_ax.axes.text(-.01, -.58, 'Contradictory', ha='right', fontsize=15, color = 'red',  fontweight='bold')
    exp_ax.axes.set_xlim(- max(np.abs(exp_df.weights)), max(np.abs(exp_df.weights)))
    exp_ax.yaxis.set_tick_params(labelsize=15)
    exp_ax.yaxis.set_tick_params(length=0)
    sns.despine(ax=exp_ax, left=True)
    
    plt.savefig(out_path, bbox_inches='tight', dpi='figure', frameon=True, edgecolor='black')

if __name__ == 'main':
    
    df = pd.DataFrame({'labels':['Lungs status=congested',
    'RR=18/min','Headache=yes','pO2=89mmHg',
    '# prior episodes of pneumonia=0','Glu=82mg/dL'], 
    'weights':[.1, -.07, .08, .02, -.09, .08]})

    draw_exp_plot(df, .95, "../output/output.png")