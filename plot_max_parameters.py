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

DF_max_params = pd.read_csv('Data/tackler_max_params_FINAL.csv')

DF_max_params.loc[DF_max_params['position'].eq('CB') | DF_max_params['position'].eq('SS') | DF_max_params['position'].eq('FS'),'position'] = 'DB'
DF_max_params.loc[DF_max_params['position'].eq('MLB') | DF_max_params['position'].eq('OLB') | DF_max_params['position'].eq('ILB'),'position'] = 'LB'
DF_max_params.loc[DF_max_params['position'].eq('DT') | DF_max_params['position'].eq('NT'),'position'] = 'T'

plt.figure()
sns.histplot(data = DF_max_params, x = 'max_vel', hue = 'position',element = 'step', kde=True)
plt.figure()
sns.histplot(data = DF_max_params, x = 'max_accel', hue = 'position',element = 'step', kde=True)

DF_max_params_DB = DF_max_params.loc[DF_max_params['position'].eq('DB')]
DF_max_params_DE = DF_max_params.loc[DF_max_params['position'].eq('DE')]
DF_max_params_LB = DF_max_params.loc[DF_max_params['position'].eq('LB')]
DF_max_params_T = DF_max_params.loc[DF_max_params['position'].eq('T')]

DF_stats_DB = DF_max_params_DB.describe()
DF_stats_DE = DF_max_params_DE.describe()
DF_stats_LB = DF_max_params_LB.describe()
DF_stats_T = DF_max_params_T.describe()

xcorr_DB = DF_max_params_DB['max_vel'].corr(DF_max_params_DB['max_accel'])
xcorr_DE = DF_max_params_DE['max_vel'].corr(DF_max_params_DE['max_accel'])
xcorr_LB = DF_max_params_LB['max_vel'].corr(DF_max_params_LB['max_accel'])
xcorr_T = DF_max_params_T['max_vel'].corr(DF_max_params_T['max_accel'])

sns_plot = sns.pairplot(DF_max_params[['position','max_vel','max_accel']], hue='position',kind='kde',corner=True)
fig = sns_plot.fig
fig.savefig('Figures/histograms_max_params')
