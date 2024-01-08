# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:59:39 2024

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
plt.close('all')

#%%
def calculate_player_stats(DF_YAP_position):
    player_ID = DF_YAP_position['NFL_ID'].unique()
    DF_store = pd.DataFrame(columns=['NFL_ID','name','position','YAP_mean','YAP_med','YAP_max','count'])
    for i_loop in range(len(player_ID)):
        DF_i = DF_YAP_position.loc[DF_YAP_position['NFL_ID']==player_ID[i_loop]].reset_index()
        DF_player_info_i = DF_i[['NFL_ID','name','position']].drop_duplicates()
        DF_stats_i = DF_i.describe()
        DF_store_i = pd.DataFrame({'NFL_ID':DF_player_info_i['NFL_ID'],'name':DF_player_info_i['name'],'position':DF_player_info_i['position'],'YAP_mean':DF_stats_i.loc['mean','YAP'],'YAP_med':DF_stats_i.loc['50%','YAP'],'YAP_max':DF_stats_i.loc['max','YAP'],'count':DF_stats_i.loc['count','YAP']})
        DF_store = pd.concat([DF_store,DF_store_i],ignore_index=True)
    
    # DF_store = DF_store.loc[DF_store['count']>10]
    return DF_store

#%%
DF_YAP = pd.read_csv('Data/tackler_YAP_FINAL.csv')
DF_max_params = pd.read_csv('Data/tackler_max_params_FINAL.csv')
DF_max_params_opt = pd.read_csv('Data/tackler_max_params_opt_FINAL.csv')

DF_max_params_opt.rename(columns={'max_vel': 'max_vel_opt', 'max_accel': 'max_accel_opt'}, inplace=True)
DF_max_params = DF_max_params.merge(DF_max_params_opt, how='left').drop_duplicates()
DF_max_params['delta_vel'] = DF_max_params['max_vel_opt']-DF_max_params['max_vel']
DF_max_params['delta_accel'] = DF_max_params['max_accel_opt']-DF_max_params['max_accel']


DF_YAP = DF_YAP.merge(DF_max_params, how='left').drop_duplicates()
DF_YAP.loc[ DF_YAP['YAP']<0 , 'YAP'] = 0

DF_YAP.loc[DF_YAP['position'].eq('CB') | DF_YAP['position'].eq('SS') | DF_YAP['position'].eq('FS'),'position'] = 'DB'
DF_YAP.loc[DF_YAP['position'].eq('MLB') | DF_YAP['position'].eq('OLB') | DF_YAP['position'].eq('ILB'),'position'] = 'LB'
DF_YAP.loc[DF_YAP['position'].eq('DT') | DF_YAP['position'].eq('NT'),'position'] = 'T'


fig = plt.figure()
sns.histplot(data = DF_YAP, x = 'YAP', hue = 'position',element = 'step', kde=False)
fig.savefig('Figures/histogram_YAP')
plt.xlim(0,10)
plt.ylim(0,500)
fig.savefig('Figures/histogram_YAP_zoomed')

sns_plot = sns.pairplot(DF_YAP[['position','max_vel_opt','max_accel_opt']], hue='position',kind='kde',corner=True)
fig = sns_plot.fig
fig.savefig('Figures/histogram_max_params_opt')


DF_YAP_DB = DF_YAP.loc[DF_YAP['position'].eq('DB')]
DF_YAP_DE = DF_YAP.loc[DF_YAP['position'].eq('DE')]
DF_YAP_LB = DF_YAP.loc[DF_YAP['position'].eq('LB')]
DF_YAP_T = DF_YAP.loc[DF_YAP['position'].eq('T')]

DF_stats_DB = DF_YAP_DB.describe()
DF_stats_DE = DF_YAP_DE.describe()
DF_stats_LB = DF_YAP_LB.describe()
DF_stats_T = DF_YAP_T.describe()

#%%
DF_player_stats = calculate_player_stats(DF_YAP_LB)
DF_player_stats = DF_player_stats.sort_values(by='count',ascending=False,ignore_index=True)
DF_player_stats_keep = DF_player_stats.loc[DF_player_stats['count']>=50]
DF_player_stats_keep = DF_player_stats_keep.sort_values(by='YAP_mean',ascending=False,ignore_index=True)

fig_dims = (6.0, 8.0)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(data=DF_player_stats_keep, y="name",x="YAP_mean",order=DF_player_stats_keep.sort_values(by='YAP_mean',ascending=True,ignore_index=True).name,color='steelblue',orient='h')
plt.tight_layout()
fig.savefig('Figures/YAP_LB')


