# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:44:33 2022

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.integrate import solve_ivp
from datetime import datetime
import imageio
pd.set_option('display.max_columns', None)
plt.close('all')


#%% Functions

def sweep_S(t,s,A,B_inv,F,G):
    #This function calculates the initial S matrix by sweeping backwards in 
    #time. This matrix is required for the initial conditions for lambda.
    
    #Convert s vector into the matrix and ensure it is symmetric.
    S = np.reshape(s,A.shape)
    S = (1.0/2)*(S+np.transpose(S))
    
    #Calculate S_dot
    SF = S@F
    SG = S@G
    S_dot = -SF-SF.T-A+SG@B_inv@SG.T
    
    #Convert matrix into vector -- the negative sign before S_dot is for 
    #integrating backwards in time
    s_dot = np.reshape(-S_dot,-1)
    
    return s_dot

def solve_ODE(t,y,A,B_inv,F,G):
    #This function integrates the ODE that includes the state variables -- 
    #position and velocity -- and lambda
    
    #Build the matrix required to to calculate the state velocities
    A_top = np.concatenate((F,-(G@B_inv@G.T)),axis = 1)
    A_bot = np.concatenate((-A,-F.T),axis = 1)
    A = np.concatenate((A_top,A_bot),axis = 0)
    
    #Calculate the state velocities
    y_dot = A@y
    
    return y_dot

def solve_optimal_path(x_0,y_0,u_0,v_0,t,V_max,A_max,R_t):
    #Calculate objective weights
    c_R = 2.0/((R_t)**2)
    beta_A = 2.0/((A_max)**2)
    alpha_V = 2.0/((V_max)**2)
    #Assign weights to the various matrix components
    c_x = c_R
    c_y = c_R
    c_u = 0.0
    c_v = 0.0
    alpha_x = 0.0
    alpha_y = 0.0
    alpha_u = alpha_V
    alpha_v = alpha_V
    beta_x = beta_A
    beta_y = beta_A
    
    #Build time vector
    # t = np.linspace(0.0,t_f,101)
    t_f = t[-1]
    
    # Build matrices
    S_f = np.diag([c_x,c_y,c_u,c_v])
    A = np.diag([alpha_x,alpha_y,alpha_u,alpha_v])
    B = np.diag([beta_x,beta_y])
    F = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    G = np.array([[0,0],[0,0],[1,0],[0,1]])
    B_inv = np.linalg.inv(B)

    # Solve for S
    s_f = np.reshape(S_f,-1)
    sol = solve_ivp(sweep_S,[0,t_f],s_f,args = (A,B_inv,F,G))
    
    #Calculate initial lambda
    s_0 = sol.y[:,-1]
    S_0 = np.reshape(s_0,S_f.shape)
    x_0 = np.array([[x_0],[y_0],[u_0],[v_0]])
    l_0 = S_0@x_0
    
    #Solve for the state variables
    xl_0 = np.concatenate((x_0,l_0),axis=0)
    sol = solve_ivp(solve_ODE,[0,t_f],xl_0[:,0],args = (A,B_inv,F,G),t_eval = t)

    #Extract state variables
    x = sol.y[0,:]
    y = sol.y[1,:]
    u = sol.y[2,:]
    v = sol.y[3,:]
    lam = sol.y[4::,:]

    #Solve for the accelerations
    acc = -B_inv@(G.T@lam)
    acc_x = acc[0,:]
    acc_y = acc[1,:]
    
    return x, y, u, v, acc_x, acc_y
    
def identify_events(DF_PBP):
    idx_start = DF_PBP.index[DF_PBP['event'].eq('handoff') | DF_PBP['event'].eq('pass_outcome_caught') | DF_PBP['event'].eq('run') | DF_PBP['event'].eq('snap_direct')]
    if len(idx_start) != 0:
        idx_start = idx_start[0].item()
    else:
        idx_start = DF_PBP.index[DF_PBP['event'].eq('ball_snap')][0].item()
    # idx_contact = DF_PBP.index[DF_PBP['event'].eq('first_contact')].item()
    idx_stop = DF_PBP.index[DF_PBP['event'].eq('tackle') | DF_PBP['event'].eq('out_of_bounds') | DF_PBP['event'].eq('fumble') | DF_PBP['event'].eq('qb_slide') | DF_PBP['event'].eq('touchdown') | DF_PBP['event'].eq('safety') | DF_PBP['event'].eq('fumble_defense_recovered')][0].item()
    
    return idx_start, idx_stop#, idx_contact
    
#%% Classes
class YAP:
    def __init__(self,game_ID,play_ID,R_t = 1.0):
        self.game_ID = game_ID
        self.play_ID = play_ID
        self.R_t = R_t
        
    def load_play(self):
        #Read in data
        DF_players = pd.read_csv('Data/players.csv')
        DF_games_all = pd.read_csv('Data/games.csv')
        DF_plays =  pd.read_csv('Data/plays.csv')
        DF_tackles =  pd.read_csv('Data/tackles.csv')

        #Reduce data down to the specified game and play
        DF_game = DF_games_all.loc[DF_games_all['gameId']==self.game_ID].reset_index()
        DF_plays = DF_plays.loc[(DF_plays['gameId']==self.game_ID) & (DF_plays['playId']==self.play_ID)].reset_index()
        DF_tackles = DF_tackles.loc[(DF_tackles['gameId']==self.game_ID) & (DF_tackles['playId']==self.play_ID)].reset_index()
        
        #Identify season/week of game
        season = DF_game.season.iloc[0]
        week = DF_game.week.iloc[0]
        self.season = season
        self.week = week
        
        #Read in play-by-play data
        DF_PBP_week = pd.read_csv('Data/tracking_week_'+str(week)+'.csv')
        DF_PBP_week = DF_PBP_week.merge(DF_players.loc[:, ['nflId', 'position']], how='left')
        DF_PBP_game = DF_PBP_week.loc[DF_PBP_week['gameId']==self.game_ID]
        DF_PBP_play = DF_PBP_game.loc[DF_PBP_game['playId']==self.play_ID]
        self.DF_PBP_play = DF_PBP_play

        #Identify yard markers
        yard_line_no = DF_plays.yardlineNumber.iloc[0]
        yards_to_go = DF_plays.yardsToGo.iloc[0]
        yard_line_side = DF_plays.yardlineSide.iloc[0]
        possession_team = DF_plays.possessionTeam.iloc[0]
        defensive_team = DF_plays.defensiveTeam.iloc[0]
        play_result = DF_plays.playResult.iloc[0]
        pre_penalty_play_result = DF_plays.prePenaltyPlayResult.iloc[0]
        play_nullified_by_penalty = DF_plays.playNullifiedByPenalty.iloc[0]
        
        self.yard_line_no = yard_line_no
        self.yards_to_go = yards_to_go
        self.yard_line_side = yard_line_side
        self.possession_team = possession_team
        self.defensive_team = defensive_team
        self.pre_penalty_play_result = pre_penalty_play_result
        self.play_result = play_result
        self.play_nullified_by_penalty = play_nullified_by_penalty
        self.play_direction = DF_PBP_play['playDirection'].head(1).item()
        
        #Identify ball carrier
        ball_carrier_ID = DF_plays.ballCarrierId.iloc[0]
        ball_carrier_name = DF_players.loc[(DF_players['nflId']==ball_carrier_ID)].displayName.iloc[0]
        ball_carrier_position = DF_players.loc[(DF_players['nflId']==ball_carrier_ID)].position.iloc[0]
        DF_PBP_ball_carrier = DF_PBP_play.loc[DF_PBP_play['nflId']==ball_carrier_ID].reset_index()
        self.ball_carrier_info = [{'NFL_ID':ball_carrier_ID,'name':ball_carrier_name,'position':ball_carrier_position,'DF_PBP':DF_PBP_ball_carrier}]
        
        #Identify tacklers
        self.DF_tackles = DF_tackles
        tackler_info = []
        for i_loop in range(len(DF_tackles)):
            tackler_ID_i = DF_tackles.nflId.iloc[i_loop]
            tackler_name_i = DF_players.loc[(DF_players['nflId']==tackler_ID_i)].displayName.iloc[0]
            tackler_position_i = DF_players.loc[(DF_players['nflId']==tackler_ID_i)].position.iloc[0]
            DF_PBP_tackler_i = DF_PBP_play.loc[DF_PBP_play['nflId']==tackler_ID_i].reset_index()
            tackler_info.append({'NFL_ID':tackler_ID_i,'name':tackler_name_i,'position':tackler_position_i,'DF_PBP':DF_PBP_tackler_i})
        self.tackler_info = tackler_info
        
    def calculate_max_params(self,save_data = False,optimal_path = False):
        DF_PBP_ball_carrier = self.ball_carrier_info[0]['DF_PBP']
        x_ball_carrier = DF_PBP_ball_carrier['x'].to_numpy()
        y_ball_carrier = DF_PBP_ball_carrier['y'].to_numpy()
        idx_events = identify_events(DF_PBP_ball_carrier)
        idx_start = idx_events[0]
        idx_stop = idx_events[1]
        
        tackler_info = self.tackler_info
        for i_loop in range(len(tackler_info)):
            #Extract data
            tackler_info_i = tackler_info[i_loop]
            if optimal_path:
                DF_PBP_tackler_i = tackler_info_i['DF_PBP_opt']
            else:
                DF_PBP_tackler_i = tackler_info_i['DF_PBP']

                
            x_tackler_i = DF_PBP_tackler_i['x'].to_numpy()
            y_tackler_i = DF_PBP_tackler_i['y'].to_numpy()
            V_tackler_i = DF_PBP_tackler_i['s'].to_numpy()
            A_tackler_i = DF_PBP_tackler_i['a'].to_numpy()
            
            #Calculate relative distance between taackle and ball carrier
            delta_x_i = x_ball_carrier-x_tackler_i
            delta_y_i = y_ball_carrier-y_tackler_i
            delta_dist_i = np.sqrt(delta_x_i**2+delta_y_i**2)
            
            #Find when tackler first gets in vicinity of ball carrier
            idx_1_i = np.nonzero(delta_dist_i < self.R_t)[0]
            if idx_1_i.size != 0:
                idx_12_i = np.nonzero(idx_1_i>idx_start)[0]  
                if idx_12_i.size != 0:
                    idx_stop_i = idx_1_i[idx_12_i[0]]   
                
                    #Find maximum value of tackler's velocity and acceleration from when the ball carrier receives the ball and when they are first in the vicinity of the ball carrier
                    tackler_V_max_i = np.max(V_tackler_i[idx_start:idx_stop_i])
                    tackler_A_max_i = np.max(A_tackler_i[idx_start:idx_stop_i])   
                else:
                    tackler_V_max_i = float('NaN')
                    tackler_A_max_i = float('NaN')
            else: 
                tackler_V_max_i = float('NaN')
                tackler_A_max_i = float('NaN')
            
            if optimal_path:
                tackler_info_i['V_max_opt'] = round(tackler_V_max_i,2)
                tackler_info_i['A_max_opt'] = round(tackler_A_max_i,2)
            else:
                tackler_info_i['V_max'] = tackler_V_max_i
                tackler_info_i['A_max'] = tackler_A_max_i
            
            if save_data:
                
                if optimal_path:
                    data_for_DF_i = {'game_ID':[self.game_ID],'play_ID':[self.play_ID],'NFL_ID':[tackler_info_i['NFL_ID']],'name':[tackler_info_i['name']],'position':[tackler_info_i['position']],'max_vel':[tackler_info_i['V_max_opt']],'max_accel':[tackler_info_i['A_max_opt']]}
                    save_name = 'Data/tackler_max_params_opt.csv'
                else:
                    data_for_DF_i = {'game_ID':[self.game_ID],'play_ID':[self.play_ID],'NFL_ID':[tackler_info_i['NFL_ID']],'name':[tackler_info_i['name']],'position':[tackler_info_i['position']],'max_vel':[tackler_info_i['V_max']],'max_accel':[tackler_info_i['A_max']]}
                    save_name = 'Data/tackler_max_params.csv'
                
                DF_i = pd.DataFrame(data=data_for_DF_i)

                if os.path.isfile(save_name):
                    DF_i.to_csv(save_name,mode='a',index=False,header=False)
                else:
                    DF_i.to_csv(save_name,mode='w',index=False,header=True)
     
        
    def animate_play(self,optimal_path = False,save_animation = False):
        
        DF_PBP_play = self.DF_PBP_play
        ball_carrier_ID = self.ball_carrier_info[0]['NFL_ID']
        possession_team = self.possession_team
        defensive_team = self.defensive_team
        
        DF_PBP_ball_carrier = self.ball_carrier_info[0]['DF_PBP']
        x_ball_carrier = DF_PBP_ball_carrier['x'].to_numpy()
        y_ball_carrier = DF_PBP_ball_carrier['y'].to_numpy()
        V_ball_carrier = DF_PBP_ball_carrier['s'].to_numpy()
        A_ball_carrier = DF_PBP_ball_carrier['a'].to_numpy()
        th_ball_carrier = DF_PBP_ball_carrier['dir'].to_numpy()*(np.pi/180.0)
        th_ball_carrier = np.arctan2(np.sin(th_ball_carrier), np.cos(th_ball_carrier))*(180.0/np.pi)
        u_ball_carrier = V_ball_carrier*np.sin(th_ball_carrier*(np.pi/180.0))
        v_ball_carrier = V_ball_carrier*np.cos(th_ball_carrier*(np.pi/180.0))

        fig = plt.figure(figsize=(10,6),layout = 'constrained')
        gs0 = fig.add_gridspec(1,2)
        gs00 = gs0[0].subgridspec(4,1)
        gs01 = gs0[1].subgridspec(1,1)
        ax = []
        for i_loop in range(gs00.nrows+1):
            if i_loop < gs00.nrows:
                ax.append(fig.add_subplot(gs00[i_loop]))
            else:
                ax.append(fig.add_subplot(gs01[0]))
        
        time = ((pd.to_datetime(DF_PBP_ball_carrier['time'])-datetime.now()).dt.total_seconds()).to_numpy()
        time = time-time[0]
        
        idx_start, idx_stop = identify_events(DF_PBP_ball_carrier)

        x_min_plot = np.floor((np.min(DF_PBP_play['x'].to_numpy())-5.0)/5.0)*5.0
        x_max_plot = np.floor((np.max(DF_PBP_play['x'].to_numpy())+5.0)/5.0)*5.0
        y_min_plot = np.floor((np.min(DF_PBP_play['y'].to_numpy())-5.0)/5.0)*5.0
        y_max_plot = np.floor((np.max(DF_PBP_play['y'].to_numpy())+5.0)/5.0)*5.0
 
        
        x_tackler = []
        y_tackler = []
        V_tackler = []
        A_tackler = []  
        th_tackler = []
        delta_dist = []
         
        for i_loop in range(len(self.tackler_info)):
            if optimal_path:
                DF_PBP_tackler_i = self.tackler_info[i_loop]['DF_PBP_opt']
            else:
                DF_PBP_tackler_i = self.tackler_info[i_loop]['DF_PBP']

                
            x_tackler_i = DF_PBP_tackler_i['x'].to_numpy()
            y_tackler_i = DF_PBP_tackler_i['y'].to_numpy()
            V_tackler_i = DF_PBP_tackler_i['s'].to_numpy()
            A_tackler_i = DF_PBP_tackler_i['a'].to_numpy()
            th_tackler_i = DF_PBP_tackler_i['dir'].to_numpy()*(np.pi/180.0)
            th_tackler_i = np.arctan2(np.sin(th_tackler_i), np.cos(th_tackler_i))*(180.0/np.pi)
            u_tackler_i = V_tackler_i*np.sin(th_tackler_i*(np.pi/180.0))
            v_tackler_i = V_tackler_i*np.cos(th_tackler_i*(np.pi/180.0))
            delta_x_i = x_ball_carrier-x_tackler_i
            delta_y_i = y_ball_carrier-y_tackler_i
            delta_dist_i = np.sqrt(delta_x_i**2+delta_y_i**2)
            
            x_tackler.append(x_tackler_i)
            y_tackler.append(y_tackler_i)
            V_tackler.append(V_tackler_i)
            A_tackler.append(A_tackler_i)
            th_tackler.append(th_tackler_i)
            delta_dist.append(delta_dist_i)
            

            
        line_style = ['-','--','-.',':']
        ax_str = ['Dist','Vel','Acc','Th']              
        i_loop = 0
        for frame_ID in DF_PBP_play['frameId'].unique():
            
            nan_check = []
            for j_loop in range(len(self.tackler_info)):
                nan_check.append(np.isnan(x_tackler[j_loop][i_loop]))      
            if all(nan_check):
                break
            
            DF_frame_i = DF_PBP_play.loc[DF_PBP_play['frameId']==frame_ID]
            
            for j_loop in range(len(ax)):
                ax[j_loop].clear()
                if j_loop < len(ax)-1:
                    ax[j_loop].axvline(x = time[idx_start], color = 'k')
                    # ax[j_loop].axvline(x = time[idx_contact], color = 'k')
                    ax[j_loop].axvline(x = time[idx_stop], color = 'k')
                    ax[j_loop].axvline(x = time[i_loop], color = 'b')
                    ax[j_loop].set_ylabel(ax_str[j_loop])
                    ax[j_loop].grid(True)

            x_ball_carrier_i = x_ball_carrier[i_loop]
            y_ball_carrier_i = y_ball_carrier[i_loop]
            th_ball_carrier_i = th_ball_carrier[i_loop]
            u_ball_carrier_i = np.sin(th_ball_carrier_i*(np.pi/180.0))
            v_ball_carrier_i = np.cos(th_ball_carrier_i*(np.pi/180.0))
            
            ax[4].set_xticks(np.arange(10,120,10))
            ax[4].set_xticklabels(['0','10','20','30','40','50','40','30','20','10','0'])
            ax[4].set_yticks([0.0+(70+3.0/4.0)*(1.0/3.0),53.3-(70+3.0/4.0)*(1.0/3.0)])
            ax[4].set_yticklabels(['',''])
            ax[4].set(xlim=(x_min_plot,x_max_plot),ylim=(y_min_plot,y_max_plot))
            ax[4].grid(True)
            ax[4].plot(DF_frame_i.loc[DF_frame_i['club']==possession_team]['x'],DF_frame_i.loc[DF_frame_i['club']==possession_team]['y'],'bs',markersize=8)
            ax[4].plot(DF_frame_i.loc[DF_frame_i['club']==defensive_team]['x'],DF_frame_i.loc[DF_frame_i['club']==defensive_team]['y'],'rs',markersize=8)
            ax[4].plot(x_ball_carrier_i,y_ball_carrier_i,'bs',markeredgecolor='k',markeredgewidth=1,markersize=8)
            ax[4].quiver(x_ball_carrier_i,y_ball_carrier_i,u_ball_carrier_i,v_ball_carrier_i)
            ax[4].plot(x_ball_carrier[0:i_loop+1],y_ball_carrier[0:i_loop+1],'b')
            ax[4].plot(DF_frame_i.loc[DF_frame_i['club']=='football']['x'],DF_frame_i.loc[DF_frame_i['club']=='football']['y'],'o',color='brown',markersize=8)
            
            for j_loop in range(len(self.tackler_info)):  
                ax[0].plot(time,delta_dist[j_loop],'r',linestyle=line_style[j_loop])
                ax[0].plot(time,self.R_t*np.ones(len(time)),'k--')
                # ax[0].plot(time[i_loop],delta_dist[i_loop],'rs',markeredgecolor='k',markeredgewidth=1,markersize=8)
                # ax[0].grid(True)
                ax[0].set(xlim=(time[0],time[-1]))
                ax[1].plot(time,V_tackler[j_loop],'r',linestyle=line_style[j_loop])
                # ax[1].plot(time[i_loop],V_tackler[i_loop],'rs',markeredgecolor='k',markeredgewidth=1,markersize=8)
                # ax[1].grid(True)
                ax[1].set(xlim=(time[0],time[-1]))
                ax[2].plot(time,A_tackler[j_loop],'r',linestyle=line_style[j_loop])
                # ax[2].plot(time[i_loop],A_tackler[i_loop],'rs',markeredgecolor='k',markeredgewidth=1,markersize=8)
                # ax[2].grid(True)
                ax[2].set(xlim=(time[0],time[-1]))
                ax[3].plot(time,th_tackler[j_loop],'r',linestyle=line_style[j_loop])
                # ax[3].grid(True)
                # ax[3].plot(time[i_loop],th_tackler[i_loop],'rs',markeredgecolor='k',markeredgewidth=1,markersize=8)
                ax[3].set(xlim=(time[0],time[-1]))
                
                x_tackler_ij = x_tackler[j_loop][i_loop]
                y_tackler_ij = y_tackler[j_loop][i_loop]
                th_tackler_ij = th_tackler[j_loop][i_loop]
                u_tackler_ij = np.sin(th_tackler_ij*(np.pi/180.0))
                v_tackler_ij = np.cos(th_tackler_ij*(np.pi/180.0))
                ax[4].quiver(x_tackler_ij,y_tackler_ij,u_tackler_ij,v_tackler_ij)
                ax[4].plot(x_tackler_ij,y_tackler_ij,'rs',markeredgecolor='k',markeredgewidth=1,markersize=8)
                ax[4].plot(x_tackler[j_loop][0:i_loop+1],y_tackler[j_loop][0:i_loop+1],'r',linestyle=line_style[j_loop])

            if save_animation:
                if optimal_path:
                    save_name = 'Figures/Animations/Optimal/Frame_'+str(i_loop).zfill(3)
                else:
                    save_name = 'Figures/Animations/Actual/Frame_'+str(i_loop).zfill(3)
                fig.savefig(save_name)       
                    
            plt.pause(0.05)
            
            i_loop += 1
        
        if save_animation:
            if optimal_path:
                save_path = 'Figures/Animations/Optimal'
            else:
                save_path = 'Figures/Animations/Actual'  
            
            images = []
            file_names = os.listdir(save_path)    
            for file_name in file_names:
                file_path = os.path.join(save_path, file_name)
                images.append(imageio.imread(file_path))
            imageio.mimsave(save_path+'/Animation.gif', images, duration = 160)

            
    def calculate_YAP(self,save_data=False):
        DF_PBP_ball_carrier = self.ball_carrier_info[0]['DF_PBP']
        ball_carrier_ID = self.ball_carrier_info[0]['NFL_ID']
        play_direction = self.play_direction
        R_f_c = self.R_t

        #Extract ball carrier data
        x_ball_carrier = DF_PBP_ball_carrier['x'].to_numpy()
        y_ball_carrier = DF_PBP_ball_carrier['y'].to_numpy()
        V_ball_carrier = DF_PBP_ball_carrier['s'].to_numpy()
        A_ball_carrier = DF_PBP_ball_carrier['a'].to_numpy()
        th_ball_carrier = DF_PBP_ball_carrier['dir'].to_numpy()*(np.pi/180.0)
        th_ball_carrier = np.arctan2(np.sin(th_ball_carrier), np.cos(th_ball_carrier))*(180.0/np.pi)
        u_ball_carrier = V_ball_carrier*np.sin(th_ball_carrier*(np.pi/180.0))
        v_ball_carrier = V_ball_carrier*np.cos(th_ball_carrier*(np.pi/180.0))
        time = (pd.to_datetime(DF_PBP_ball_carrier['time'])-datetime.now()).dt.total_seconds()
        time = time-time[0]
        
        #Identify events
        idx_start, idx_stop = identify_events(DF_PBP_ball_carrier)
        time_0 = time[idx_start]

        #Extract tackler data
        for i_loop in range(len(self.tackler_info)):
            tackler_info_i = self.tackler_info[i_loop]
            DF_PBP_tackler_i = tackler_info_i['DF_PBP']
            tackler_position_i = tackler_info_i['position']
            if (tackler_position_i == 'CB') | (tackler_position_i == 'FS') | (tackler_position_i == 'SS') | (tackler_position_i == 'DB'):
                V_max_med = 5.6
                V_max_max = 10.9
                A_max_med = 4.2
                A_max_max = 9.1
            elif (tackler_position_i == 'DE'):
                V_max_med = 3.7
                V_max_max = 9.7
                A_max_med = 2.8
                A_max_max = 6.2
            elif (tackler_position_i == 'MLB') | (tackler_position_i == 'OLB') | (tackler_position_i == 'ILB'):
                V_max_med = 5.0
                V_max_max = 10.7
                A_max_med = 3.9
                A_max_max = 9.4
            elif (tackler_position_i == 'DT') | (tackler_position_i == 'NT'):
                V_max_med = 3.2
                V_max_max = 8.8
                A_max_med = 2.4
                A_max_max = 7.1
                
            x_tackler_i = DF_PBP_tackler_i['x'].to_numpy()
            y_tackler_i = DF_PBP_tackler_i['y'].to_numpy()
            V_tackler_i = DF_PBP_tackler_i['s'].to_numpy()
            A_tackler_i = DF_PBP_tackler_i['a'].to_numpy()
            th_tackler_i = DF_PBP_tackler_i['dir'].to_numpy()*(np.pi/180.0)
            th_tackler_i = np.arctan2(np.sin(th_tackler_i), np.cos(th_tackler_i))*(180.0/np.pi)
            u_tackler_i = V_tackler_i*np.sin(th_tackler_i*(np.pi/180.0))
            v_tackler_i = V_tackler_i*np.cos(th_tackler_i*(np.pi/180.0))
            delta_x_i = x_ball_carrier-x_tackler_i
            delta_y_i = y_ball_carrier-y_tackler_i
            delta_dist_i = np.sqrt(delta_x_i**2+delta_y_i**2)       
       
            tackler_info_i['DF_PBP_opt'] = DF_PBP_tackler_i.copy()
            tackler_info_i['YAP'] = float('NaN')
            
            #Find when tackler first gets in vacinity of ball carrier
            idx_1_i = np.nonzero(delta_dist_i < self.R_t)[0]
            if (idx_1_i.size != 0):
                idx_12_i = np.nonzero(idx_1_i>idx_start)[0]    
                if (idx_12_i.size != 0):
                    idx_i = idx_1_i[idx_12_i[0]]    
                    x_ball_carrier_v_i = x_ball_carrier[idx_i]
                     
                    x_tackler_0_i = x_tackler_i[idx_start]
                    y_tackler_0_i = y_tackler_i[idx_start]
                    u_tackler_0_i = u_tackler_i[idx_start]
                    v_tackler_0_i = v_tackler_i[idx_start]   
                
        
                    for j_loop in range(len(time[idx_start+1:-1])):
                        idx_j = idx_start+1+j_loop
                        time_j = time[idx_start:idx_j+1].to_numpy()-time_0
                        x_ball_carrier_j = x_ball_carrier[idx_j]
                        y_ball_carrier_j = y_ball_carrier[idx_j]
                        delta_x_0_ij = x_tackler_0_i-x_ball_carrier_j
                        delta_y_0_ij = y_tackler_0_i-y_ball_carrier_j
                        delta_dist_0_ij = np.sqrt(delta_x_0_ij**2+delta_y_0_ij**2)
                        
                        x_path_ij, y_path_ij, u_path_ij, v_path_ij, acc_x_path_ij, acc_y_path_ij = solve_optimal_path(delta_x_0_ij,delta_y_0_ij,u_tackler_0_i,v_tackler_0_i,time_j,V_max_med,A_max_med,R_f_c)
                        R_f_ij = np.sqrt(x_path_ij[-1]**2+y_path_ij[-1]**2)
                        V_max_ij = np.max(np.sqrt(u_path_ij**2+v_path_ij**2))
                        A_max_ij = np.max(np.sqrt(acc_x_path_ij**2+acc_y_path_ij**2))
                        
                        if (R_f_ij <= R_f_c) & (V_max_ij <= V_max_max) & (A_max_ij <= A_max_max):
                            x_opt_i = x_path_ij+x_ball_carrier_j
                            y_opt_i = y_path_ij+y_ball_carrier_j
                            u_opt_i = u_path_ij
                            v_opt_i = v_path_ij
                            V_opt_i = np.sqrt(u_opt_i**2+v_opt_i**2)
                            acc_x_opt_i = acc_x_path_ij
                            acc_y_opt_i = acc_y_path_ij
                            A_opt_i = np.sqrt(acc_x_opt_i**2+acc_y_opt_i**2)
                            th_opt_i = np.arctan2(u_opt_i,v_opt_i)*180.0/np.pi
                            DF_PBP_tackler_opt_i = DF_PBP_tackler_i.copy()
                            idx_1 = idx_start
                            idx_2 = idx_start+len(x_opt_i)-1
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'x'] = x_opt_i
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'y'] = y_opt_i
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'s'] = V_opt_i
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'a'] = A_opt_i
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'dir'] = th_opt_i
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'o'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'dis'] = float('NaN')
        
                            
                            idx_1 = idx_2+1
                            idx_2 = len(DF_PBP_tackler_opt_i)
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'x'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'y'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'s'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'a'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'dir'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'o'] = float('NaN')
                            DF_PBP_tackler_opt_i.loc[idx_1:idx_2,'dis'] = float('NaN')
        
                            if play_direction == 'left':
                                YAP_i = x_ball_carrier_j-x_ball_carrier_v_i
                            else:
                                YAP_i = x_ball_carrier_v_i-x_ball_carrier_j
                            
                            tackler_info_i['DF_PBP_opt'] = DF_PBP_tackler_opt_i
                            tackler_info_i['YAP'] = round(YAP_i,2)
                            break  
                
            if save_data:
                data_for_DF_i = {'game_ID':[self.game_ID],'play_ID':[self.play_ID],'NFL_ID':[tackler_info_i['NFL_ID']],'name':[tackler_info_i['name']],'position':[tackler_info_i['position']],'YAP':[tackler_info_i['YAP']]}
                save_name = 'Data/tackler_YAP.csv'
                DF_i = pd.DataFrame(data=data_for_DF_i)
    
                if os.path.isfile(save_name):
                    DF_i.to_csv(save_name,mode='a',index=False,header=False)
                else:
                    DF_i.to_csv(save_name,mode='w',index=False,header=True)
