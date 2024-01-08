
import pandas as pd
import matplotlib.pyplot as plt
import optimal_pursuit as op

pd.set_option('display.max_columns', None)
plt.close('all')



#%% User Inputs

game_ID = 2022091102
play_ID = 800

#%%
Y1 = op.YAP(game_ID,play_ID)
Y1.load_play()
Y1.animate_play(save_animation = True)
Y1.calculate_YAP()
Y1.animate_play(optimal_path=True,save_animation = True)
Y1.calculate_max_params()
Y1.calculate_max_params(optimal_path=True)
