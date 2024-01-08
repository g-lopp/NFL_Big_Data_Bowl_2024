# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:44:33 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import optimal_pursuit as op
import time
import os

pd.set_option('display.max_columns', None)
plt.close('all')

#%%

DF_tackles = pd.read_csv('Data/tackles.csv')
DF_game_and_play = DF_tackles[['gameId','playId']]
DF_game_and_play = DF_game_and_play.drop_duplicates().reset_index()

time_0 = time.time()
# run_err_list = pd.read_csv('Data/run_errors_YAP.csv',header=None)[0].tolist()

save_name = 'Data/run_errors_YAP.csv'

for i_loop in range(len(DF_game_and_play)):

    game_ID_i = DF_game_and_play.at[i_loop,'gameId'].item()
    play_ID_i = DF_game_and_play.at[i_loop,'playId'].item()
  
    # YAP_i = op.YAP(game_ID_i,play_ID_i)
    # YAP_i.load_play()
    # YAP_i.calculate_YAP(save_data=True)
    # YAP_i.calculate_max_params(save_data=True,optimal_path=True)
    try:
        YAP_i = op.YAP(game_ID_i,play_ID_i)
        YAP_i.load_play()
        YAP_i.calculate_YAP(save_data=True)
        YAP_i.calculate_max_params(save_data=True,optimal_path=True)
    except:
        data_for_DF_i = {'game_ID':[game_ID_i],'play_ID':[play_ID_i],'run_number':[i_loop]}
        DF_i = pd.DataFrame(data=data_for_DF_i)

        if os.path.isfile(save_name):
            DF_i.to_csv(save_name,mode='a',index=False,header=False)
        else:
            DF_i.to_csv(save_name,mode='w',index=False,header=True)
        
    time_elapsed = time.time()-time_0
    print('Progress: Loop ' + str(i_loop) + '/' + str(len(DF_game_and_play)-1) + ' in ' + str(time_elapsed) + ' s')

