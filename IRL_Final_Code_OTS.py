
# coding: utf-8

# ##### Code to test the rolling-pin prediction based on NO OBSTACLE scenario for 2 lane and 4 lane grid. 
# ###### changes:
# ###### 1. Add prediction changes for all  forward obstacle scenarios

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math
import itertools
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler,Normalizer

#import irl.maxent as maxent
import irl.value_iteration as value_iteration
import irl.mdp.gridworld as gridworld
import irl.deep_maxent as deep_maxent
from irl.deep_maxent import normalize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from random import shuffle
from frechetdist import frdist
from sklearn.ensemble import RandomForestRegressor
import pickle
import seaborn as sns
import cv2
from collections import defaultdict


# In[2]:


def warpAffine(img,M,rows,cols):
    dst=np.zeros((rows,cols))
    for i in range(0,rows):
        dst[i][0]=M[0][0]*img[i][0]+M[0][1]*img[i][1]+M[0][2]
        dst[i][1]=M[1][0]*img[i][0]+M[1][1]*img[i][1]+M[1][2]
    return dst


# In[3]:


def determine_actions(x1,y1,x2,y2,trajectory,i):
    """
    Function computes the action taken based on the x and y co-ordinates at two consecutive time steps
    """
    if x1==x2:
        if y1==y2:
            trajectory[i, 1] = (0,0) #not moved
        elif y1<y2:
            trajectory[i, 1] = (0,1) #straight
    elif y1<y2:
        if x1<x2:
            trajectory[i, 1] = (1,1) # moved diagonally towards right
        elif x1>x2:
            trajectory[i, 1] = (-1,1) # moved diagonally towards left           
    else:
        trajectory[i, 1] = (0,0) #not moved


# In[4]:


def determine_action(x1,y1,x2,y2):
    """
    Function computes the action taken based on the x and y co-ordinates at two consecutive time steps
    """
    if x1==x2:
        if y1==y2:
            return (0,0) #not moved
        elif y1<y2:
            return (0,1) #straight            
    elif y1<y2:
        if x1<x2:
            return (1,1) # moved diagonally towards right
        elif x1>x2:
             return (-1,1) # moved diagonally towards left  
    else:
        return (0,0) #not moved


# In[5]:


def load_trajectories(gw,trajectories,df_in,i,xmin=0,ymin=0):
    """
    Read data from the csv using pandas
    """ 
    trajectory = np.zeros((1, 21), dtype=object)
    #print(df_in.shape)
    mat=df_in.as_matrix()
    #print(mat.shape)instead 
    j=0
    trajectory[j, 0] = (df_in.iloc[i]['x']), (df_in.iloc[i]['y'])##current  coord
    #print('trajectory[j, 0]',trajectory[j, 0])
    
    #print(mat.shape[0])
    if ((i+1)==df_in.shape[0]):
        trajectory[j,2] = (df_in.iloc[i]['x']), (df_in.iloc[i]['y'])#next coord
        determine_actions(df_in.iloc[i]['x'], df_in.iloc[i]['y'],df_in.iloc[i]['x'], df_in.iloc[i]['y'],trajectory,j)#action
    else:
        trajectory[j,2] = (df_in.iloc[i+1]['x']), (df_in.iloc[i+1]['y'])#next coord
        determine_actions(df_in.iloc[i]['x'], df_in.iloc[i]['x'],df_in.iloc[i+1]['x'], df_in.iloc[i+1]['y'],trajectory,j)#action
    trajectory[j,3] = (df_in.iloc[i]['Dist_F'])#rel_dist_F
    trajectory[j,4] = (df_in.iloc[i]['Delta_v_F'])#rel_F_vel
    trajectory[j,5] = (df_in.iloc[i]['Dist_F2'])#rel_dist_F2
    trajectory[j,6] = (df_in.iloc[i]['Delta_v_F2'])#rel_F2_vel
    trajectory[j,7] = (df_in.iloc[i]['Dist_B'])#rel_dist_B
    trajectory[j,8] = (df_in.iloc[i]['Delta_v_B'])#rel_vel_B
    trajectory[j,9] = (df_in.iloc[i]['Dist_B2'])#rel_dist_B2
    trajectory[j,10] = (df_in.iloc[i]['Delta_v_B2'])#rel_vel_B2
    trajectory[j,11] = (df_in.iloc[i]['Lane'])#Lane ID
                      
    trajectory[j,12] = (df_in.iloc[i]['Dist_FL'])#rel_dist_FL
    trajectory[j,13] = (df_in.iloc[i]['Delta_v_FL'])#rel_vel_FL
    trajectory[j,14] = (df_in.iloc[i]['Dist_FR'])#rel_dist_FR
    trajectory[j,15] = (df_in.iloc[i]['Delta_v_FR'])#rel_vel_FR
    trajectory[j,16] = (df_in.iloc[i]['Dist_BR'])#rel_dist_BR
    trajectory[j,17] = (df_in.iloc[i]['Delta_v_BR'])#rel_vel_BR
    trajectory[j,18] = (df_in.iloc[i]['Dist_BL'])#rel_dist_BL
    trajectory[j,19] = (df_in.iloc[i]['Delta_v_BL'])#rel_vel_BL
    trajectory[j,20] = (df_in.iloc[i]['speed'])#vel
                      
        #trajectory[i,2] = (mat[i+1][4]-xmin), (mat[i+1][5]-ymin)
    #to_draw[mat[i][4]-16][mat[i][5]-48] = \
    #        to_draw[mat[i][4]-16][mat[i][5]-48]+ 1
    
    #print(gw.point_to_int((trajectory[0, 0][0],trajectory[0,1][0])))
            
    for k in range(0, 1):#trajectory.shape[0]):
        #print(k)   
        new_traj = []
        #print(trajectory[k, 2][0], trajectory[k, 2][1])
        new_traj.append((
            (int(gw.point_to_int((trajectory[k, 0][0],trajectory[k,0][1])))),#current_state
             trajectory[k,1],#action
             #(gw.reward(int(gw.point_to_int((trajectory[k,2][0], trajectory[k, 2][1]))),final_states)),#reward for next state
             (int(gw.point_to_int((trajectory[k, 2][0],trajectory[k,2][1])))),#next_state
             trajectory[k,3],#rel_pre_x               TODO:comments to be updated
             trajectory[k,4],#rel_pre_y
             trajectory[k,5],#rel_pre_vel
             trajectory[k,6],#rel_foll_x
             trajectory[k,7],#rel_foll_y
             trajectory[k,8],#rel_foll_vel
             trajectory[k,9],#vel
             trajectory[k,10],#laneID
             trajectory[k,11],#vel
             trajectory[k,12],#vel
             trajectory[k,13],#vel
             trajectory[k,14],#vel
             trajectory[k,15],#vel
             trajectory[k,16],#vel
             trajectory[k,17],#vel
             trajectory[k,18],#vel
             trajectory[k,19],#vel
             trajectory[k,20]#vel
            ))
        
        trajectories.append(new_traj)  
    #print('trajectories:',trajectories[length-2],gw.point_to_int(final_states))


# In[6]:


def predict (mat,gw,maxent_policy_FL_FR_B1_B2_BL_m1,maxent_policy_FL_FR_B1_B2_BL_m2,maxent_policy_F1_F2_FL_BL_BR_m1,maxent_policy_F1_F2_FL_BL_BR_m2,maxent_policy_F1_F2_FL_FR_BL_BR_m1,maxent_policy_F1_F2_FL_FR_BL_BR_m2,
maxent_policy_FL_FR_B1_B2_BL_BR_m1,maxent_policy_FL_FR_B1_B2_BL_BR_m2,maxent_policy_F1_FR_B1_B2_BL_BR_m1,maxent_policy_F1_FR_B1_B2_BL_BR_m2,
maxent_policy_F1_FR_B1_BL_BR_m1,maxent_policy_F1_FR_B1_BL_BR_m2,
maxent_policy_F1_F2_FR_B1_BL_m1,maxent_policy_F1_F2_FR_B1_BL_m2,
maxent_policy_F1_F2_FR_B1_BL_BR_m1,maxent_policy_F1_F2_FR_B1_BL_BR_m2,maxent_policy_F1_F2_FL_FR_B1_BL_m1,maxent_policy_F1_F2_FL_FR_B1_BL_m2,maxent_policy_F1_FL_FR_B1_BL_m1,maxent_policy_F1_FL_FR_B1_BL_m2,maxent_policy_F1_FL_FR_B1_B2_BL_m1,maxent_policy_F1_FL_FR_B1_B2_BL_m2,maxent_policy_F1_FR_B1_BR_m1,maxent_policy_F1_FR_B1_BR_m2,maxent_policy_F1_F2_FR_B1_BR_m1,maxent_policy_F1_F2_FR_B1_BR_m2,maxent_policy_F1_F2_FR_BR_m1,maxent_policy_F1_F2_FR_BR_m2,maxent_policy_FR_B1_B2_BL_m1,maxent_policy_FR_B1_B2_BL_m2,maxent_policy_F1_FR_B1_BL_m1,maxent_policy_F1_FR_B1_BL_m2,maxent_policy_F1_FR_B1_B2_m1,maxent_policy_F1_FR_B1_B2_m2,maxent_policy_F1_FL_FR_B1_B2_m1,maxent_policy_F1_FL_FR_B1_B2_m2,maxent_policy_F1_FL_FR_B1_B2_BR_m1,maxent_policy_F1_FL_FR_B1_B2_BR_m2,maxent_policy_F1_FL_FR_B1_BR_m1,maxent_policy_F1_FL_FR_B1_BR_m2,maxent_policy_F1_F2_FL_B1_BL_BR_m1,maxent_policy_F1_F2_FL_B1_BL_BR_m2,maxent_policy_FL_B1_B2_BL_m1,maxent_policy_FL_B1_B2_BL_m2,maxent_policy_F1_F2_B1_B2_BL_m1,maxent_policy_F1_F2_B1_B2_BL_m2,
maxent_policy_F1_FL_FR_B1_B2_BL_BR_m1,maxent_policy_F1_FL_FR_B1_B2_BL_BR_m2,maxent_policy_F1_F2_FL_FR_BR_m1,maxent_policy_F1_F2_FL_FR_BR_m2,maxent_policy_F1_F2_FL_BL_m1,maxent_policy_F1_F2_FL_BL_m2,maxent_policy_F1_F2_FL_B1_BL_m1,maxent_policy_F1_F2_FL_B1_BL_m2,maxent_policy_F1_FL_B1_BL_m1,maxent_policy_F1_FL_B1_BL_m2,
maxent_policy_F1_F2_FL_B1_m1,maxent_policy_F1_F2_FL_B1_m2,maxent_policy_F1_B1_B2_BL_m1,maxent_policy_F1_B1_B2_BL_m2,maxent_policy_F1_F2_FL_BR_m1,maxent_policy_F1_F2_FL_BR_m2,maxent_policy_F1_F2_FL_B1_BR_m1,maxent_policy_F1_F2_FL_B1_BR_m2,maxent_policy_F1_F2_FL_FR_B1_BR_m1,maxent_policy_F1_F2_FL_FR_B1_BR_m2,
maxent_policy_F1_F2_FL_FR_B1_B2_BR_m1,maxent_policy_F1_F2_FL_FR_B1_B2_BR_m2,maxent_policy_F1_F2_FL_FR_B1_BL_BR_m1,maxent_policy_F1_F2_FL_FR_B1_BL_BR_m2,maxent_policy_F1_F2_FL_B1_B2_BL_m1,maxent_policy_F1_F2_FL_B1_B2_BL_m2,
maxent_policy_F1_FL_B1_B2_BL_m1,maxent_policy_F1_FL_B1_B2_BL_m2,
maxent_policy_FR_BL_BR_m1,maxent_policy_FR_BL_BR_m2,maxent_policy_FR_B1_BL_BR_m1,maxent_policy_FR_B1_BL_BR_m2,maxent_policy_FR_B1_B2_BL_BR_m1,maxent_policy_FR_B1_B2_BL_BR_m2,
               maxent_policy_FR_BR_m1,maxent_policy_FR_BR_m2,maxent_policy_FR_BL_m1,maxent_policy_FR_BL_m2,
               maxent_policy_FR_B1_BR_m1,maxent_policy_FR_B1_BR_m2,maxent_policy_FR_B1_BL_m1,maxent_policy_FR_B1_BL_m2,
                maxent_policy_FR_B1_m1,maxent_policy_FR_B1_B2_m2,maxent_policy_FR_B1_B2_m1,maxent_policy_FR_B1_m2,
maxent_policy_FL_BL_BR_m1,maxent_policy_FL_BL_BR_m2,maxent_policy_FL_B1_BL_BR_m1,maxent_policy_FL_B1_BL_BR_m2,maxent_policy_FL_B1_B2_BL_BR_m1,maxent_policy_FL_B1_B2_BL_BR_m2,
               maxent_policy_FL_BR_m1,maxent_policy_FL_BR_m2,maxent_policy_FL_BL_m1,maxent_policy_FL_BL_m2,
               maxent_policy_FL_B1_BR_m1,maxent_policy_FL_B1_BR_m2,maxent_policy_FL_B1_BL_m1,maxent_policy_FL_B1_BL_m2,
                maxent_policy_FL_B1_m1,maxent_policy_FL_B1_B2_m2,maxent_policy_FL_B1_B2_m1,maxent_policy_FL_B1_m2,
maxent_policy_F1_F2_BL_BR_m1,maxent_policy_F1_F2_BL_BR_m2,maxent_policy_F1_F2_B1_BL_BR_m1,maxent_policy_F1_F2_B1_BL_BR_m2,maxent_policy_F1_F2_B1_B2_BL_BR_m1,maxent_policy_F1_F2_B1_B2_BL_BR_m2,
               maxent_policy_F1_F2_BR_m1,maxent_policy_F1_F2_BR_m2,maxent_policy_F1_F2_BL_m1,maxent_policy_F1_F2_BL_m2,
               maxent_policy_F1_F2_B1_BR_m1,maxent_policy_F1_F2_B1_BR_m2,maxent_policy_F1_F2_B1_BL_m1,maxent_policy_F1_F2_B1_BL_m2,
                maxent_policy_F1_F2_B1_m1,maxent_policy_F1_F2_B1_B2_m2,maxent_policy_F1_F2_B1_B2_m1,maxent_policy_F1_F2_B1_m2,maxent_policy_F1_BL_BR_m1,maxent_policy_F1_BL_BR_m2,maxent_policy_F1_B1_BL_BR_m1,maxent_policy_F1_B1_BL_BR_m2,maxent_policy_F1_B1_B2_BL_BR_m1,maxent_policy_F1_B1_B2_BL_BR_m2,
               maxent_policy_F1_BR_m1,maxent_policy_F1_BR_m2,maxent_policy_F1_BL_m1,maxent_policy_F1_BL_m2,
               maxent_policy_F1_B1_BR_m1,maxent_policy_F1_B1_BR_m2,maxent_policy_F1_B1_BL_m1,maxent_policy_F1_B1_BL_m2,
                maxent_policy_F1_B1_m1,maxent_policy_F1_B1_B2_m2,maxent_policy_F1_B1_B2_m1,maxent_policy_F1_B1_m2,
                maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m1,maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m2,maxent_policy_B1_B2_BR_BL_m1,maxent_policy_B1_B2_BR_BL_m2,
                maxent_policy_B1_BR_BL_m1,maxent_policy_B1_BR_BL_m2, maxent_policy_B1_BL_B2_m1,maxent_policy_B1_BL_B2_m2,
                maxent_policy_B1_BR_B2_m1,maxent_policy_B1_BR_B2_m2,maxent_policy_B1_BL_m1,
                maxent_policy_B1_BL_m2,maxent_policy_BL_BR_m1,maxent_policy_BL_BR_m2,
                maxent_policy_B1_m1,maxent_policy_B1_m2,maxent_policy_B1_B2_m1,maxent_policy_B1_B2_m2,
                maxent_policy_B1_BR_m1,maxent_policy_B1_BR_m2,maxent_policy_BL_m1,
                maxent_policy_BL_m2, maxent_policy_BR_m1,maxent_policy_BR_m2,
                maxent_policy_F1_F2_FR_FL_m1,maxent_policy_F1_F2_FR_FL_m2,
                maxent_policy_F1_FR_FL_m1,maxent_policy_F1_FR_FL_m2, maxent_policy_F1_FL_F2_m1,maxent_policy_F1_FL_F2_m2,
                maxent_policy_F1_FR_F2_m1,maxent_policy_F1_FR_F2_m2,maxent_policy_F1_FL_m1,
                maxent_policy_F1_FL_m2,maxent_policy_FL_FR_m1,maxent_policy_FL_FR_m2,
                maxent_policy_F1_m1,maxent_policy_F1_m2,maxent_policy_F1_F2_m1,maxent_policy_F1_F2_m2,
                maxent_policy_F1_FR_m1,maxent_policy_F1_FR_m2,maxent_policy_m1,maxent_policy_m2,maxent_policy_FL_m1,
                maxent_policy_FL_m2, maxent_policy_FR_m1,maxent_policy_FR_m2,
             current_state, to_draw_result_x,to_draw_result_y, x, y,n_traj,t_id,speed,actual_speed):
    to_draw_result_x[0] =(x)
    to_draw_result_y[0] = (y)
    time=0.12
    j=0
    m2=False
    for i in range(1,n_traj):
        ID_F=mat[j][15]
        ID_F2=mat[j][21]
        ID_B=mat[j][27]
        ID_B2=mat[j][33]
        ID_FL=mat[j][39]
        ID_BL=mat[j][45]
        ID_BR=mat[j][51]
        ID_FR=mat[j][57]
        j=j+3

        print("ID_F: {0}, ID_F2 : {1}, ID_B: {2}, ID_B2 : {3},ID_FL: {4}, ID_BL : {5}, ID_BR: {6}, ID_FR : {7} ".format(ID_F,ID_F2, ID_B, ID_B2,ID_FL,ID_BL,ID_BR, ID_FR))
        if (x<8 and y<25):
            if ((ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_FL_m1
                print('maxent_policy_F1_FR_FL_m1 used')
            elif ((ID_F and ID_FL and ID_FR and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_F2_FR_FL_m1
                print('maxent_policy_F1_F2_FR_FL_m1 used')
            elif ((ID_F and ID_FR and ID_F2 and ID_BR) and not (ID_B or ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_F2_FR_BR_m1
                print('maxent_policy_F1_F2_FR_BR_m1 used')
            elif ((ID_F and ID_FR and ID_F2 and ID_BR and ID_B) and not (ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_F2_FR_B1_BR_m1
                print('maxent_policy_F1_F2_FR_B1_BR_m1 used')
            elif ((ID_F and ID_FR and ID_BR and ID_B) and not (ID_F2 or ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_FR_B1_BR_m1
                print('maxent_policy_F1_FR_B1_BR_m1 used')
            elif ((ID_F and ID_FL and ID_F2) and not (ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_F2_m1
                print('maxent_policy_F1_FL_F2_m1 used')
            elif ((ID_F and ID_F2 and ID_FR) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_F2_m1
                print('maxent_policy_F1_FR_F2_m1 used')
            elif ((ID_F and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                policy = maxent_policy_F1_F2_m1
                print('maxent_policy_F1_F2_m1 used')
            elif ((ID_F and ID_FR) and not (ID_FL or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_m1
                print('maxent_policy_F1_FR_m1 used')
            elif ((ID_FL and ID_FR) and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_FL_FR_m1
                print('maxent_policy_FL_FR_m1 used')
            elif ((ID_F and ID_FL) and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_m1
                print('maxent_policy_F1_FL_m1 used')
            elif (ID_FL and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                policy = maxent_policy_FL_m1
                print('maxent_policy_FL_m1 used')
            elif (ID_FR and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_FR_m1
                print('maxent_policy_FR_m1 used')
            elif (ID_F and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_F1_m1
                print('maxent_policy_F1_m1 used')
###################################################################################################
            elif ((ID_B and ID_BL and ID_BR) and not (ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_BL_m1
                print('maxent_policy_B1_BR_BL_m1 used')
            elif ((ID_B and ID_BL and ID_BR and ID_B2) and not (ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_B2_BR_BL_m1
                print('maxent_policy_B1_B2_BR_BL_m1 used')
            elif ((ID_B and ID_BL and ID_B2) and not (ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BL_B2_m1
                print('maxent_policy_B1_BL_B2_m1 used')
            elif ((ID_B and ID_B2 and ID_BR) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_B2_m1
                print('maxent_policy_B1_BR_B2_m1 used')
            elif ((ID_B and ID_B2) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                policy = maxent_policy_B1_B2_m1
                print('maxent_policy_B1_B2_m1 used')
            elif ((ID_B and ID_BR) and not (ID_BL or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_m1
                print('maxent_policy_B1_BR_m1 used')
            elif ((ID_BL and ID_BR) and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_BL_BR_m1
                print('maxent_policy_BL_BR_m1 used')
            elif ((ID_B and ID_BL) and not (ID_BR or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BL_m1
                print('maxent_policy_B1_BL_m1 used')
            elif (ID_BL and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                policy = maxent_policy_BL_m1
                print('maxent_policy_BL_m1 used')
            elif (ID_BR and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_BR_m1
                print('maxent_policy_BR_m1 used')
            elif (ID_B and not (ID_BR or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_B1_m1
                print('maxent_policy_B1_m1 used')
            elif (ID_B and ID_BR and ID_B2 and ID_F and ID_F2 and ID_FL and ID_FR and ID_BL):
                policy = maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m1
                print('maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m1 used')
            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR and ID_BL) and not (ID_F2)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BL_BR_m1
                print('maxent_policy_F1_FL_FR_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B2 or ID_BL)):
                policy = maxent_policy_F1_FL_FR_B1_BR_m1
                print('maxent_policy_F1_FL_FR_B1_BR_m1 used')
            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR and ID_B2) and not (ID_F2 or ID_BL)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BR_m1
                print('maxent_policy_F1_FL_FR_B1_B2_BR_m1 used')
            elif ((ID_B and ID_F and ID_FL and ID_FR and ID_B2) and not (ID_F2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_B2_m1
                print('maxent_policy_F1_FL_FR_B1_B2_m1 used')
            elif ((ID_B and ID_F  and ID_FR and ID_B2) and not (ID_F2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_F1_FR_B1_B2_m1
                print('maxent_policy_F1_FR_B1_B2_m1 used')
            elif ((ID_B and ID_F  and ID_FR and ID_BL) and not (ID_F2 or ID_B2 or ID_BR or ID_FL)):
                policy = maxent_policy_F1_FR_B1_BL_m1
                print('maxent_policy_F1_FR_B1_BL_m1 used')
            elif ((ID_B and ID_FR and ID_BL and ID_B2) and not (ID_F or ID_F2 or ID_BR or ID_FL)):
                policy = maxent_policy_FR_B1_B2_BL_m1
                print('maxent_policy_FR_B1_B2_BL_m1 used')
            elif ((ID_B and ID_F) and not(ID_BR or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_m1
                print('maxent_policy_F1_B1_m1 used')
            elif ((ID_BL and ID_F) and not(ID_BR or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_BL_m1
                print('maxent_policy_F1_BL_m1 used')
            elif ((ID_BR and ID_F) and not(ID_BL or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_BR_m1
                print('maxent_policy_F1_BR_m1 used')
            elif ((ID_B and ID_F and ID_B2) and not(ID_BR or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_B2_m1
                print('maxent_policy_F1_B1_B2_m1 used')
            elif ((ID_B and ID_F and ID_BR) and not(ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_BR_m1
                print('maxent_policy_F1_B1_BR_m1 used')
            elif ((ID_B and ID_F and ID_BL) and not(ID_BR or ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_B1_BL_m1
                print('maxent_policy_F1_B1_BL_m1 used')
            elif ((ID_BR and ID_F and ID_BL) and not(ID_B or ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_BL_BR_m1
                print('maxent_policy_F1_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_B1_BL_BR_m1
                print('maxent_policy_F1_B1_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_F1_B1_B2_BL_BR_m1
                print('maxent_policy_F1_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_B2 and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR and ID_BR)):
                policy = maxent_policy_F1_B1_B2_BL_m1
                print('maxent_policy_F1_B1_B2_BL_m1 used')
############################################################################################# FL combination
            elif ((ID_B and ID_FL) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_m1
                print('maxent_policy_FL_B1_m1 used')
            elif ((ID_BL and ID_FL) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FR or ID_B)):
                policy = maxent_policy_FL_BL_m1
                print('maxent_policy_FL_BL_m1 used')
            elif ((ID_BR and ID_FL) and not(ID_BL or ID_B2  or ID_F2 or ID_F or ID_FR or ID_B)):
                policy = maxent_policy_FL_BR_m1
                print('maxent_policy_FL_BR_m1 used')
            elif ((ID_B and ID_FL and ID_B2) and not(ID_BR or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_B2_m1
                print('maxent_policy_FL_B1_B2_m1 used')
            elif ((ID_B and ID_FL and ID_BR) and not(ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_BR_m1
                print('maxent_policy_FL_B1_BR_m1 used')
            elif ((ID_B and ID_FL and ID_BL) and not(ID_BR or ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_B1_BL_m1
                print('maxent_policy_FL_B1_BL_m1 used')
            elif ((ID_BR and ID_FL and ID_BL) and not(ID_B or ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_BL_BR_m1
                print('maxent_policy_FL_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_B1_BL_BR_m1
                print('maxent_policy_FL_B1_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR)):
                policy = maxent_policy_FL_B1_B2_BL_BR_m1
                print('maxent_policy_FL_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_B2 and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR or ID_BR)):
                policy = maxent_policy_FL_B1_B2_BL_m1
                print('maxent_policy_FL_B1_B2_BL_m1 used')
############################################################################################## FR combination
            elif ((ID_B and ID_FR) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_m1
                print('maxent_policy_FR_B1_m1 used')
            elif ((ID_BL and ID_FR) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FL or ID_B)):
                policy = maxent_policy_FR_BL_m1
                print('maxent_policy_FR_BL_m1 used')
            elif ((ID_BR and ID_FR) and not(ID_BL or ID_B2  or ID_F2 or ID_F or ID_FL or ID_B)):
                policy = maxent_policy_FR_BR_m1
                print('maxent_policy_FR_BR_m1 used')
            elif ((ID_B and ID_FR and ID_B2) and not(ID_BR or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_B2_m1
                print('maxent_policy_FR_B1_B2_m1 used')
            elif ((ID_B and ID_FR and ID_BR) and not(ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_BR_m1
                print('maxent_policy_FR_B1_BR_m1 used')
            elif ((ID_B and ID_FR and ID_BL) and not(ID_BR or ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_B1_BL_m1
                print('maxent_policy_FR_B1_BL_m1 used')
            elif ((ID_BR and ID_FR and ID_BL) and not(ID_B or ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_BL_BR_m1
                print('maxent_policy_FR_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_FR and ID_BL) and not(ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_B1_BL_BR_m1
                print('maxent_policy_FR_B1_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_FR and ID_BL) and not(ID_F2 or ID_F or ID_FL)):
                policy = maxent_policy_FR_B1_B2_BL_BR_m1
                print('maxent_policy_FR_B1_B2_BL_BR_m1 used')
############
            elif ((ID_B and ID_F and ID_F2 and ID_FR and ID_BL and ID_BR) and not(ID_B2 or ID_FL)):
                policy = maxent_policy_F1_F2_FR_B1_BL_BR_m1
                print('maxent_policy_F1_F2_FR_B1_BL_BR_m1 used')
            elif ((ID_B and ID_F and ID_F2 and ID_FR and ID_BL) and not(ID_B2 or ID_FL or ID_BR)):
                policy = maxent_policy_F1_F2_FR_B1_BL_m1
                print('maxent_policy_F1_F2_FR_B1_BL_m1 used')
            elif ((ID_B and ID_F and ID_BR and ID_FR and ID_BL) and not(ID_B2 or ID_FL or ID_F2)):
                policy = maxent_policy_F1_FR_B1_BL_BR_m1
                print('maxent_policy_F1_FR_B1_BL_BR_m1 used')
            elif ((ID_B and ID_F and ID_BR and ID_FR and ID_BL and ID_B2) and not( ID_FL or ID_F2)):
                policy = maxent_policy_F1_FR_B1_B2_BL_BR_m1
                print('maxent_policy_F1_FR_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_FL and ID_BR and ID_FR and ID_BL and ID_B2) and not( ID_F or ID_F2)):
                policy = maxent_policy_FL_FR_B1_B2_BL_BR_m1
                print('maxent_policy_FL_FR_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_FL and ID_FR and ID_BL and ID_B2) and not( ID_F or ID_F2 or ID_BR)):
                policy = maxent_policy_FL_FR_B1_B2_BL_m1
                print('maxent_policy_FL_FR_B1_B2_BL_m1 used')
            elif ((ID_FL and ID_BL and ID_BR and ID_F and ID_F2 ) and not(ID_B or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_FL_BL_BR_m1
                print('maxent_policy_F1_F2_FL_BL_BR_m1 used')
            elif ((ID_FL and ID_BL and ID_BR and ID_F and ID_F2 and ID_FR) and not(ID_B or ID_B2)):
                policy = maxent_policy_F1_F2_FL_FR_BL_BR_m1
                print('maxent_policy_F1_F2_FL_FR_BL_BR_m1 used')
############################################################################################ID_F1_F2 combination
            elif ((ID_B and ID_F and ID_F2) and not(ID_BR or ID_B2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_m1
                print('maxent_policy_F1_F2_B1_m1 used')
            elif ((ID_BL and ID_F and ID_F2) and not(ID_BR or ID_B2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_F2_BL_m1
                print('maxent_policy_F1_F2_BL_m1 used')
            elif ((ID_BR and ID_F and ID_F2) and not(ID_BL or ID_B2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_F2_BR_m1
                print('maxent_policy_F1_F2_BR_m1 used')
            elif ((ID_B and ID_F and ID_B2 and ID_F2) and not(ID_BR or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_B2_m1
                print('maxent_policy_F1_F2_B1_B2_m1 used')
            elif ((ID_B and ID_F and ID_BR and ID_F2) and not(ID_B2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_BR_m1
                print('maxent_policy_F1_F2_B1_BR_m1 used')
            elif ((ID_B and ID_F and ID_BL and ID_F2) and not(ID_BR or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_B1_BL_m1
                print('maxent_policy_F1_F2_B1_BL_m1 used')
            elif ((ID_BR and ID_F and ID_BL and ID_F2) and not(ID_B or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_BL_BR_m1
                print('maxent_policy_F1_F2_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_B1_BL_BR_m1
                print('maxent_policy_F1_F2_B1_BL_BR_m1 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR)):
                policy = maxent_policy_F1_F2_B1_B2_BL_BR_m1
                print('maxent_policy_F1_F2_B1_B2_BL_BR_m1 used')
            elif ((ID_B and ID_B2 and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_B1_B2_BL_m1
                print('maxent_policy_F1_F2_B1_B2_BL_m1 used')
##########################################################################################################
            elif ((ID_F and ID_BR and ID_F2 and ID_FL) and not(ID_B or ID_B2 or ID_BL or ID_FR)):
                policy = maxent_policy_F1_F2_FL_BR_m1
                print('maxent_policy_F1_F2_FL_BR_m1 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL) and not(ID_B2 or ID_BL or ID_FR)):
                policy = maxent_policy_F1_F2_FL_B1_BR_m1
                print('maxent_policy_F1_F2_FL_B1_BR_m1 used')
            elif ((ID_F and ID_B and ID_F2 and ID_FL) and not(ID_B2 or ID_BL or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_B1_m1
                print('maxent_policy_F1_F2_FL_B1_m1 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL) and not(ID_B2 or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_B1_BL_m1
                print('maxent_policy_F1_F2_FL_B1_BL_m1 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL and ID_BR) and not(ID_B2 or ID_FR)):
                policy = maxent_policy_F1_F2_FL_B1_BL_BR_m1
                print('maxent_policy_F1_F2_FL_B1_BL_BR_m1 used')
            elif ((ID_F and ID_B and ID_BL and  ID_FL) and not(ID_B2 or ID_FR or ID_BR or ID_F2)):
                policy = maxent_policy_F1_FL_B1_BL_m1
                print('maxent_policy_F1_FL_B1_BL_m1 used')
            elif ((ID_F and ID_BL and ID_F2 and ID_FL) and not(ID_B2 or ID_B or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_BL_m1
                print('maxent_policy_F1_F2_FL_BL_m1 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BL)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BR_m1
                print('maxent_policy_F1_F2_FL_FR_B1_BR_m1 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BR)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BL_m1
                print('maxent_policy_F1_F2_FL_FR_B1_BL_m1 used')
            elif ((ID_F and ID_B and ID_BL and ID_FL and ID_FR) and not(ID_F2 or ID_B2 or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_BL_m1
                print('maxent_policy_F1_FL_FR_B1_BL_m1 used')
            elif ((ID_F and ID_B and ID_BL and ID_FL and ID_FR and ID_B2) and not(ID_F2 or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BL_m1
                print('maxent_policy_F1_FL_FR_B1_B2_BL_m1 used')
            elif ((ID_F and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BL or ID_B)):
                policy = maxent_policy_F1_F2_FL_FR_BR_m1
                print('maxent_policy_F1_F2_FL_FR_BR_m1 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_BL)):
                policy = maxent_policy_F1_F2_FL_FR_B1_B2_BR_m1
                print('maxent_policy_F1_F2_FL_FR_B1_B2_BR_m1 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL and ID_FR and ID_BL) and not(ID_B2)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BL_BR_m1
                print('maxent_policy_F1_F2_FL_FR_B1_BL_BR_m1 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_F2 and ID_FL) and not(ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_FR_B1_B2_BL_m1
                print('maxent_policy_F1_F2_FL_FR_B1_B2_BL_m1 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_FL) and not(ID_FR or ID_BR or ID_F2)):
                policy = maxent_policy_F1_FL_B1_B2_BL_m1
                print('maxent_policy_F1_FL_B1_B2_BL_m1 used')
#########################################################################################################
            else:
                policy = maxent_policy_m1
                print('maxent_policy_m1 used')
        else:
            m2=True
            if ((ID_F and ID_FR and ID_FL) and not (ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_FL_m2
                print('maxent_policy_F1_FR_FL_m2 used')
            elif ((ID_F and ID_FR and ID_FL and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_F2_FR_FL_m2
                print('maxent_policy_F1_F2_FR_FL_m2 used')
            elif ((ID_F and ID_F2 and ID_FL) and not (ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_F2_m2
                print('maxent_policy_F1_FL_F2_m2 used')
            elif ((ID_F and ID_F2 and ID_FR) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_F2_m2
                print('maxent_policy_F1_FR_F2_m2 used')
            elif ((ID_B and ID_FL and ID_FR and ID_BL and ID_B2) and not( ID_F or ID_F2 or ID_BR)):
                policy = maxent_policy_FL_FR_B1_B2_BL_m2
                print('maxent_policy_FL_FR_B1_B2_BL_m2 used')
            elif ((ID_FL and ID_BL and ID_BR and ID_F and ID_F2 ) and not(ID_B or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_FL_BL_BR_m2
                print('maxent_policy_F1_F2_FL_BL_BR_m2 used')
            elif ((ID_FL and ID_BL and ID_BR and ID_F and ID_F2 and ID_FR) and not(ID_B or ID_B2)):
                policy = maxent_policy_F1_F2_FL_FR_BL_BR_m2
                print('maxent_policy_F1_F2_FL_FR_BL_BR_m2 used')
            elif ((ID_F and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                policy = maxent_policy_F1_F2_m2
                print('maxent_policy_F1_F2_m2 used')
            elif ((ID_F and ID_FR) and not (ID_FL or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FR_m2
                print('maxent_policy_FL_FR_m2 used')
            elif ((ID_FL and ID_FR) and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_FL_FR_m2
                print('maxent_policy_FL_FR_m2 used')
            elif ((ID_F and ID_FL) and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_m2
                print('maxent_policy_F1_FL_m2 used')
            elif (ID_FL and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                policy = maxent_policy_FL_m2
                print('maxent_policy_FL_m2 used')
            elif (ID_FR and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_FR_m2
                print('maxent_policy_FR_m2 used')
            elif (ID_F and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_F1_m2
                print('maxent_policy_F1_m2 used')
############################################################################################################
            elif ((ID_B and ID_BR and ID_BL) and not (ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_BL_m2
                print('maxent_policy_B1_BR_BL_m2 used')
            elif ((ID_B and ID_BR and ID_BL and ID_B2) and not ( ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_B2_BR_BL_m2
                print('maxent_policy_B1_B2_BR_BL_m2 used')
            elif ((ID_B and ID_B2 and ID_BL) and not (ID_BR or  ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BL_B2_m2
                print('maxent_policy_B1_BL_B2_m2 used')
            elif ((ID_B and ID_B2 and ID_BR) and not (ID_BL or  ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_B2_m2
                print('maxent_policy_B1_BR_B2_m2 used')
            elif ((ID_B and ID_B2) and not (ID_BL or ID_B or  ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_B2_m2
                print('maxent_policy_B1_B2_m2 used')
            elif ((ID_B and ID_BR) and not (ID_BL or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BR_m2
                print('maxent_policy_BL_BR_m2 used')
            elif ((ID_BL and ID_BR) and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_BL_BR_m2
                print('maxent_policy_BL_BR_m2 used')
            elif ((ID_B and ID_BL) and not (ID_BR or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_B1_BL_m2
                print('maxent_policy_B1_BL_m2 used')
            elif (ID_BL and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                policy = maxent_policy_BL_m2
                print('maxent_policy_BL_m2 used')
            elif (ID_BR and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_BR_m2
                print('maxent_policy_BR_m2 used')
            elif (ID_B and not (ID_BR or ID_B2 or  ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_B1_m2
                print('maxent_policy_B1_m2 used')
            elif (ID_B and ID_BR and ID_B2 and ID_F and ID_F2 and ID_FL and ID_FR and ID_BL):
                policy = maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m2
                print('maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m2 used')
            elif ((ID_B and ID_F) and not(ID_BR or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_m2
                print('maxent_policy_F1_B1_m2 used')
            elif ((ID_BL and ID_F) and not(ID_BR or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_BL_m2
                print('maxent_policy_F1_BL_m2 used')
            elif ((ID_BR and ID_F) and not(ID_BL or ID_B2  or ID_F2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_BR_m2
                print('maxent_policy_F1_BR_m2 used')
            elif ((ID_B and ID_F and ID_B2) and not(ID_BR or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_B2_m2
                print('maxent_policy_F1_B1_B2_m2 used')
            elif ((ID_B and ID_F and ID_BR) and not(ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_B1_BR_m2
                print('maxent_policy_F1_B1_BR_m2 used')
            elif ((ID_B and ID_F and ID_BL) and not(ID_BR or ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_B1_BL_m2
                print('maxent_policy_F1_B1_BL_m2 used')
            elif ((ID_BR and ID_F and ID_BL) and not(ID_B or ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_BL_BR_m2
                print('maxent_policy_F1_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_B1_BL_BR_m2
                print('maxent_policy_F1_B1_BL_BR_m2 used')
            elif ((ID_B and ID_B2 and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR and ID_BR)):
                policy = maxent_policy_F1_B1_B2_BL_m2
                print('maxent_policy_F1_B1_B2_BL_m2 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_F and ID_BL) and not(ID_F2 or ID_FL or ID_FR)):
                policy = maxent_policy_F1_B1_B2_BL_BR_m2
                print('maxent_policy_F1_B1_B2_BL_BR_m2 used')
            elif ((ID_B and ID_F and ID_F2) and not(ID_BR or ID_B2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_m1
                print('maxent_policy_F1_F2_B1_m1 used')
            elif ((ID_BL and ID_F and ID_F2) and not(ID_BR or ID_B2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_F2_BL_m2
                print('maxent_policy_F1_F2_BL_m2 used')
            elif ((ID_BR and ID_F and ID_F2) and not(ID_BL or ID_B2 or ID_FL or ID_FR or ID_B)):
                policy = maxent_policy_F1_F2_BR_m2
                print('maxent_policy_F1_F2_BR_m2 used')
            elif ((ID_B and ID_F and ID_B2 and ID_F2) and not(ID_BR or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_B2_m2
                print('maxent_policy_F1_F2_B1_B2_m2 used')
            elif ((ID_B and ID_F and ID_BR and ID_F2) and not(ID_B2 or ID_FL or ID_FR or ID_BL)):
                policy = maxent_policy_F1_F2_B1_BR_m2
                print('maxent_policy_F1_F2_B1_BR_m2 used')
            elif ((ID_B and ID_F and ID_BL and ID_F2) and not(ID_BR or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_B1_BL_m2
                print('maxent_policy_F1_F2_B1_BL_m2 used')
            elif ((ID_BR and ID_F and ID_BL and ID_F2) and not(ID_B or ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_BL_BR_m2
                print('maxent_policy_F1_F2_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR or ID_B2)):
                policy = maxent_policy_F1_F2_B1_BL_BR_m2
                print('maxent_policy_F1_F2_B1_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR)):
                policy = maxent_policy_F1_F2_B1_B2_BL_BR_m2
                print('maxent_policy_F1_F2_B1_B2_BL_BR_m2 used')
            elif ((ID_B and ID_FL) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_m2
                print('maxent_policy_FL_B1_m2 used')
            elif ((ID_BL and ID_FL) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FR or ID_B)):
                policy = maxent_policy_FL_BL_m2
                print('maxent_policy_FL_BL_m2 used')
            elif ((ID_BR and ID_FL) and not(ID_BL or ID_B2  or ID_F2 or ID_F or ID_FR or ID_B)):
                policy = maxent_policy_FL_BR_m2
                print('maxent_policy_FL_BR_m2 used')
            elif ((ID_B and ID_FL and ID_B2) and not(ID_BR or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_B2_m2
                print('maxent_policy_FL_B1_B2_m2 used')
            elif ((ID_B and ID_FL and ID_BR) and not(ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                policy = maxent_policy_FL_B1_BR_m2
                print('maxent_policy_FL_B1_BR_m2 used')
            elif ((ID_B and ID_FL and ID_BL) and not(ID_BR or ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_B1_BL_m2
                print('maxent_policy_FL_B1_BL_m2 used')
###########
            elif ((ID_B and ID_F and ID_F2 and ID_FR and ID_BL and ID_BR) and not(ID_B2 or ID_FL)):
                policy = maxent_policy_F1_F2_FR_B1_BL_BR_m2
                print('maxent_policy_F1_F2_FR_B1_BL_BR_m2 used')
            elif ((ID_B and ID_F and ID_F2 and ID_FR and ID_BL) and not(ID_B2 or ID_FL or ID_BR)):
                policy = maxent_policy_F1_F2_FR_B1_BL_m2
                print('maxent_policy_F1_F2_FR_B1_BL_m2 used')
            elif ((ID_B and ID_F and ID_BR and ID_FR and ID_BL) and not(ID_B2 or ID_FL or ID_F2)):
                policy = maxent_policy_F1_FR_B1_BL_BR_m2
                print('maxent_policy_F1_FR_B1_BL_BR_m2 used')
            elif ((ID_B and ID_F and ID_BR and ID_FR and ID_BL and ID_B2) and not( ID_FL or ID_F2)):
                policy = maxent_policy_F1_FR_B1_B2_BL_BR_m2
                print('maxent_policy_F1_FR_B1_B2_BL_BR_m2 used')
            elif ((ID_B and ID_FL and ID_BR and ID_FR and ID_BL and ID_B2) and not( ID_F or ID_F2)):
                policy = maxent_policy_FL_FR_B1_B2_BL_BR_m2
                print('maxent_policy_FL_FR_B1_B2_BL_BR_m2 used')
            elif ((ID_BR and ID_FL and ID_BL) and not(ID_B or ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_BL_BR_m2
                print('maxent_policy_FL_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR or ID_B2)):
                policy = maxent_policy_FL_B1_BL_BR_m2
                print('maxent_policy_FL_B1_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR)):
                policy = maxent_policy_FL_B1_B2_BL_BR_m2
                print('maxent_policy_FL_B1_B2_BL_BR_m2 used')
            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR and ID_BL) and not (ID_F2)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BL_BR_m2
                print('maxent_policy_F1_FL_FR_B1_B2_BL_BR_m2 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BR)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BL_m2
                print('maxent_policy_F1_F2_FL_FR_B1_BL_m2 used')
            elif ((ID_F and ID_B and ID_BL and ID_FL and ID_FR) and not(ID_F2 or ID_B2 or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_BL_m2
                print('maxent_policy_F1_FL_FR_B1_BL_m2 used')
            elif ((ID_F and ID_B and ID_BL and ID_FL and ID_FR and ID_B2) and not(ID_F2 or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BL_m2
                print('maxent_policy_F1_FL_FR_B1_B2_BL_m2 used')
            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B2 or ID_BL)):
                policy = maxent_policy_F1_FL_FR_B1_BR_m2
                print('maxent_policy_F1_FL_FR_B1_BR_m2 used')
            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR and ID_B2) and not (ID_F2 or ID_BL)):
                policy = maxent_policy_F1_FL_FR_B1_B2_BR_m2
                print('maxent_policy_F1_FL_FR_B1_B2_BR_m2 used')
            elif ((ID_B and ID_F and ID_FL and ID_FR and ID_B2) and not (ID_F2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_FL_FR_B1_B2_m2
                print('maxent_policy_F1_FL_FR_B1_B2_m2 used')
            elif ((ID_B and ID_F  and ID_FR and ID_B2) and not (ID_F2 or ID_BL or ID_BR or ID_FL)):
                policy = maxent_policy_F1_FR_B1_B2_m2
                print('maxent_policy_F1_FR_B1_B2_m2 used')
############################################################################################## FR combination
            elif ((ID_B and ID_FR) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_m2
                print('maxent_policy_FR_B1_m2 used')
            elif ((ID_BL and ID_FR) and not(ID_BR or ID_B2  or ID_F2 or ID_F or ID_FL or ID_B)):
                policy = maxent_policy_FR_BL_m2
                print('maxent_policy_FR_BL_m2 used')
            elif ((ID_BR and ID_FR) and not(ID_BL or ID_B2  or ID_F2 or ID_F or ID_FL or ID_B)):
                policy = maxent_policy_FR_BR_m2
                print('maxent_policy_FR_BR_m2 used')
            elif ((ID_B and ID_FR and ID_B2) and not(ID_BR or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_B2_m2
                print('maxent_policy_FR_B1_B2_m2 used')
            elif ((ID_B and ID_FR and ID_BR) and not(ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                policy = maxent_policy_FR_B1_BR_m2
                print('maxent_policy_FR_B1_BR_m2 used')
            elif ((ID_B and ID_FR and ID_BL) and not(ID_BR or ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_B1_BL_m2
                print('maxent_policy_FR_B1_BL_m2 used')
            elif ((ID_BR and ID_FR and ID_BL) and not(ID_B or ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_BL_BR_m2
                print('maxent_policy_FR_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_FR and ID_BL) and not(ID_F2 or ID_F or ID_FL or ID_B2)):
                policy = maxent_policy_FR_B1_BL_BR_m2
                print('maxent_policy_FR_B1_BL_BR_m2 used')
            elif ((ID_BR and ID_B and ID_B2 and ID_FR and ID_BL) and not(ID_F2 or ID_F or ID_FL)):
                policy = maxent_policy_FR_B1_B2_BL_BR_m2
                print('maxent_policy_FR_B1_B2_BL_BR_m2 used')
#######################################################################################################
            elif ((ID_F and ID_BR and ID_F2 and ID_FL) and not(ID_B or ID_B2 or ID_BL or ID_FR)):
                policy = maxent_policy_F1_F2_FL_BR_m2
                print('maxent_policy_F1_F2_FL_BR_m2 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL) and not(ID_B2 or ID_BL or ID_FR)):
                policy = maxent_policy_F1_F2_FL_B1_BR_m2
                print('maxent_policy_F1_F2_FL_B1_BR_m2 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BL)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BR_m2
                print('maxent_policy_F1_F2_FL_FR_B1_BR_m2 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_BL)):
                policy = maxent_policy_F1_F2_FL_FR_B1_B2_BR_m2
                print('maxent_policy_F1_F2_FL_FR_B1_B2_BR_m2 used')
            elif ((ID_F and ID_B and ID_BR and ID_F2 and ID_FL and ID_FR and ID_BL) and not(ID_B2)):
                policy = maxent_policy_F1_F2_FL_FR_B1_BL_BR_m2
                print('maxent_policy_F1_F2_FL_FR_B1_BL_BR_m2 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_F2 and ID_FL) and not(ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_B1_B2_BL_m2
                print('maxent_policy_F1_F2_FL_B1_B2_BL_m2 used')
            elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_FL) and not(ID_FR or ID_BR or ID_F2)):
                policy = maxent_policy_F1_FL_B1_B2_BL_m2
                print('maxent_policy_F1_FL_B1_B2_BL_m2 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL and ID_BR) and not(ID_B2 or ID_FR)):
                policy = maxent_policy_F1_F2_FL_B1_BL_BR_m2
                print('maxent_policy_F1_F2_FL_B1_BL_BR_m2 used')
            elif ((ID_F and ID_B and ID_BL and ID_F2 and ID_FL) and not(ID_B2 or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_B1_BL_m2
                print('maxent_policy_F1_F2_FL_B1_BL_m2 used')
            elif ((ID_F and ID_B and ID_BL and  ID_FL) and not(ID_B2 or ID_FR or ID_BR or ID_F2)):
                policy = maxent_policy_F1_FL_B1_BL_m2
                print('maxent_policy_F1_FL_B1_BL_m2 used')
            elif ((ID_F and ID_B and ID_BL and  ID_B2) and not(ID_FL or ID_FR or ID_BR or ID_F2)):
                policy = maxent_policy_F1_B1_B2_BL_m2
                print('maxent_policy_F1_B1_B2_BL_m2 used')
            elif ((ID_F and ID_B and ID_F2 and ID_FL) and not(ID_B2 or ID_BL or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_B1_m2
                print('maxent_policy_F1_F2_FL_B1_m2 used')
            elif ((ID_B and ID_F  and ID_FR and ID_BL) and not (ID_F2 or ID_B2 or ID_BR or ID_FL)):
                policy = maxent_policy_F1_FR_B1_BL_m2
                print('maxent_policy_F1_FR_B1_BL_m2 used')
            elif ((ID_B and ID_FR and ID_BL and ID_B2) and not (ID_F or ID_F2 or ID_BR or ID_FL)):
                policy = maxent_policy_FR_B1_B2_BL_m2
                print('maxent_policy_FR_B1_B2_BL_m2 used')
            elif ((ID_F and ID_BR and ID_F2 and ID_FL and ID_FR) and not(ID_B2 or ID_BL or ID_B)):
                policy = maxent_policy_F1_F2_FL_FR_BR_m2
                print('maxent_policy_F1_F2_FL_FR_BR_m2 used')
            elif ((ID_B and ID_B2 and ID_F and ID_BL and ID_F2) and not(ID_FL or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_B1_B2_BL_m2
                print('maxent_policy_F1_F2_B1_B2_BL_m2 used')
            elif ((ID_B and ID_B2 and ID_FL and ID_BL) and not(ID_F2 or ID_F or ID_FR or ID_BR)):
                policy = maxent_policy_FL_B1_B2_BL_m2
                print('maxent_policy_FL_B1_B2_BL_m2 used')
            elif ((ID_F and ID_BL and ID_F2 and ID_FL) and not(ID_B2 or ID_B or ID_FR or ID_BR)):
                policy = maxent_policy_F1_F2_FL_BL_m2
                print('maxent_policy_F1_F2_FL_BL_m2 used')
            elif ((ID_F and ID_FL and ID_FR and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                policy = maxent_policy_F1_F2_FR_FL_m2
                print('maxent_policy_F1_F2_FR_FL_m2 used')
            elif ((ID_F and ID_FR and ID_F2 and ID_BR) and not (ID_B or ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_F2_FR_BR_m2
                print('maxent_policy_F1_F2_FR_BR_m2 used')
            elif ((ID_F and ID_FR and ID_F2 and ID_BR and ID_B) and not (ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_F2_FR_B1_BR_m2
                print('maxent_policy_F1_F2_FR_B1_BR_m2 used')
            elif ((ID_F and ID_FR and ID_BR and ID_B) and not (ID_F2 or ID_B2 or ID_BL or ID_FL)):
                policy = maxent_policy_F1_FR_B1_BR_m2
                print('maxent_policy_F1_FR_B1_BR_m2 used')
            else:
                policy = maxent_policy_m2
                print('maxent_policy_m2 used')
        print('i:',i)
        print('Current State:',current_state)
        print('Actual Speed:',actual_speed[i])
        print('predicted speed:',speed[i])
        action = []
        action.append(float(policy[int(current_state)][0]))
        print('Action(0,0):',action[0])
        action.append(float(policy[int(current_state)][1]))
        print('Action(0,1):',action[1])
        action.append(float(policy[int(current_state)][2]))
        print('Action(1,1):',action[2])
        action.append(float(policy[int(current_state)][3]))
        print('Action(-1,1):',action[3])
    
        
        if (action[0] == 0.25) and (action[1] == 0.25) and (action[2] == 0.25) and (action[3] == 0.25):
                y =  y + speed[i]*time #speed[i]*0.04### 40ms and speed in m/second -> 0.12
                to_draw_result_x[i] = x
                to_draw_result_y[i] =  y
                print('Action1(0,1):',x,y)
        else: 
            if action[3]>=action[0] and action[3]>=action[1] and action[3]>=action[2]:# and (action[3] >=0.40):#left lane
                #di_speed_x= (speed[i-1]+speed[i])*-0.5
                x = x- 0.1
                y = y + speed[i]*time #speed[i]*0.04### 40ms and speed in m/second
                to_draw_result_x[i] = x
                to_draw_result_y[i] = y
                print('Action3(-1,1):',x,y)
            elif action[1]>=action[0] and action[1]>=action[2] and action[1]>=action[3]:#straight
                y = y + speed[i]*time### 40ms and speed in m/second
                to_draw_result_x[i] = x
                to_draw_result_y[i] = y
                print('Action1(0,1):',x,y)
            elif action[2]>=action[0] and action[2]>=action[1] and action[2]>=action[3]:#right lane
                #di_speed= (speed[i-1]+speed[i])*0.5
                x1 = x + 0.1
                y = y + speed[i]*time### 40ms and speed in m/second
                if x1 > 15:
                    x = x - 0.1
                else:
                    x = x1
                to_draw_result_x[i] = x
                to_draw_result_y[i] = y
                print('Action2(1,1):',x,y)
            else: #stay
                to_draw_result_x[i] = x
                to_draw_result_y[i] = y
                print('Action0(0,0):',x,y)
            current_state = int(gw.point_to_int_predict((x, y),m2))
            m2=False
            #state[current_state]=0.8
    #state[final_state] = 1
    #np.savetxt('C:/Users/ga67zod/Desktop/Pooja/Results/Predicted'+str(t_id)+'.txt',np.concatenate((to_draw_result_x,to_draw_result_y),axis=0), delimiter=',')


# In[7]:


def speed(min_max_scaler):
    df = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\Features_classificationGP070239_lane_VideoNo.txt',delimiter ="\t").fillna(value = 0)
    df_x = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\x.txt',delimiter ="\t").fillna(value = 0)
    df['x']=df_x
    df_y = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\y.txt',delimiter ="\t").fillna(value = 0)
    df['y']=df_y 
    df.Dist_F[df.ID_F == 0] = 300
    df.Dist_F[df.ID_F2 == 0] = 300
    df.Dist_F[df.ID_FL == 0] = 300
    df.Dist_F[df.ID_FR == 0] = 300
    le = LabelEncoder()
    le.fit(df['ID'])
    vehicle_id = le.classes_
    vehicle_id = vehicle_id[:-1]
    split_point=0.05
    vehicle_id=vehicle_id[:int(len(vehicle_id) *0.66)]
    training_ids=vehicle_id[:int(len(vehicle_id)*0.9)]
    #test_ids = [1,44,61,63,84,92] #vehicle_id[int(len(vehicle_id)*0.9):]
    check=False
    for v_id in training_ids:
        #print(v_id)
        df_in=df.loc[df['ID']==v_id]
        ##computing speed based on the new coordiantes
        x = df_in['x'].as_matrix()
        y = df_in['y'].as_matrix()
        displacement_x=np.zeros((x.shape),dtype=np.float32)
        displacement_y=np.zeros((y.shape),dtype=np.float32)
        for i in range(1,x.shape[0]):
            displacement_x[i] = x[i]-x[i-1]
            displacement_y[i] = y[i]-y[i-1]
        #print(displacement_x,displacement_y)
        displacement = np.sqrt(np.square(displacement_x)+np.square(displacement_y))
        #print('displacement',displacement)
        new_speed = displacement/0.04
        #print('new_speed',new_speed)
        df_in['speed']=new_speed
        mat=df_in['speed'].as_matrix()  
        mat = np.delete(mat, (0), axis=0)
        new = df_in.as_matrix()
        speed_t_1=new[:, [6]]
        speed_t_1 = np.roll(speed_t_1, 1)
        new[:, [6]]=speed_t_1
        new = np.delete(new, (0), axis=0)
        if check:
            X=np.append(X,new,axis=0)
            speed=np.append(speed,mat,axis=0)
        else:
            X=new
            speed=mat
            check=True
    features = X[:][:,(6,10,17,20,23,26,41,44,53,56)]
    row,col=features.shape
    #speed = sc.fit_transform(speed.reshape((int(row),1)))
    features=min_max_scaler.fit_transform(features)   


# In[8]:


def visualize(avg_grnd_rwd,avg_reward,grid_size_x,grid_size_y,which_grid,size_x,size_y):
    fig = plt.figure(figsize=(size_x,size_y))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax = sns.heatmap(np.reshape(avg_grnd_rwd, (grid_size_x,grid_size_y), order='F'),vmin=-1, vmax=1, linewidth=0.5,annot=True,ax=ax1)
    ax.invert_yaxis()
    ax.set_title('Avg Ground Truth Reward'+which_grid)
    ax1 = sns.heatmap(np.reshape(avg_reward, (grid_size_x,grid_size_y), order='F'),vmin=-1, vmax=1, linewidth=0.5,annot=True,ax=ax2)
    ax1.invert_yaxis()
    ax1.set_title('Avg Predicted Reward'+which_grid)
    #plt.show()


# In[9]:


def computation(first_ego,avg_reward,avg_grnd_reward,deep_max,feature_matrix,gw, GAMMA, trajectories,obs_trajectories, learning_rate, epochs,scope_name,write_true = False):
    if first_ego:
            l2_loss,r = deep_max.deep_maxent_irl(avg_grnd_reward,feature_matrix,gw.transition_probability, GAMMA, trajectories,obs_trajectories, learning_rate, epochs,scope_name,write_true)
            first_ego=False
    else:
            l2_loss,r = deep_max.train(avg_grnd_reward,feature_matrix,gw.transition_probability, GAMMA, trajectories,obs_trajectories, learning_rate, epochs,write_true)                
    avg_reward = np.mean([r.ravel(),avg_reward], axis=0)
    return (avg_reward,first_ego,l2_loss)


# In[10]:


def add_vids_to_list(d ,key,value):
    #if key in d.keys():  
    d[key].add(value)  
    #else:   
    #    d[key] = value


# In[11]:


def main(grid_size_x,grid_size_y,discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """
    veh_id=[129]
    wind = 0.8
    n_iter = epochs
    decay_rate=0.9
    trajectory_length = 3*grid_size_x   
    sc = MinMaxScaler()
    f_sc = MinMaxScaler()
    border_m1 = list()
    border_m2 = list()
    for i in range(0,180,9):
        border_m1.append(i)
        border_m1.append(i+8)
    for i in range(0,280,14):
        border_m2.append(i)
        border_m2.append(i+13)
    model_filename = 'finalized_1112.sav'
    loss_FL_m1=loss_m1=loss_FL_m2=loss_m2=loss_FR_m1=loss_FR_m2=loss_F1_m1=loss_F1_m2=loss_F1_F2_m1=loss_F1_F2_m2=float("inf") 
    loss_F1_FR_m1=loss_F1_FR_m2=loss_F1_FL_m1=loss_F1_FL_m2=loss_FL_FR_m1=loss_FL_FR_m2=loss_F1_FR_F2_m1=loss_F1_FR_F2_m2=float("inf")
    loss_F1_FL_F2_m1=loss_F1_FL_F2_m2=loss_F1_FR_FL_m1=loss_F1_FR_FL_m2=loss_F1_F2_FR_FL_m1=loss_F1_F2_FR_FL_m2=float("inf")
    loss_BL_m1=loss_m1=loss_BL_m2=loss_m2=loss_BR_m1=loss_BR_m2=loss_B1_m1=loss_B1_m2=loss_B1_B2_m1=loss_B1_B2_m2=float("inf") 
    loss_B1_BR_m1=loss_B1_BR_m2=loss_B1_BL_m1=loss_B1_BL_m2=loss_BL_BR_m1=loss_BL_BR_m2=loss_B1_BR_B2_m1=loss_B1_BR_B2_m2=float("inf")
    loss_B1_BL_B2_m1=loss_B1_BL_B2_m2=loss_B1_BR_BL_m1=loss_B1_BR_BL_m2=loss_B1_B2_BR_BL_m1=loss_B1_B2_BR_BL_m2=float("inf")
    loss_F1_F2_FR_FL_B1_B2_BR_BL_m1=loss_F1_F2_FR_FL_B1_B2_BR_BL_m2=loss_F1_B1_m1=loss_F1_B1_m2=loss_F1_B1_B2_m1=loss_F1_B1_B2_m2=float('inf')
    loss_F1_B1_BL_m1=loss_F1_B1_BL_m2=loss_F1_B1_BR_m1=loss_F1_B1_BR_m2=loss_F1_BL_m2=loss_F1_BL_m1=loss_F1_BR_m2=loss_F1_BR_m1=float('inf')
    loss_F1_BL_BR_m1=loss_F1_BL_BR_m2=loss_F1_B1_BL_BR_m1=loss_F1_B1_BL_BR_m2=loss_F1_B1_B2_BL_BR_m1=loss_F1_B1_B2_BL_BR_m2=float('inf')
    loss_FR_B1_m1=loss_FR_B1_m2=loss_FR_B1_B2_m1=loss_FR_B1_B2_m2=float('inf')
    loss_FR_B1_BL_m1=loss_FR_B1_BL_m2=loss_FR_B1_BR_m1=loss_FR_B1_BR_m2=loss_FR_BL_m2=loss_FR_BL_m1=loss_FR_BR_m2=loss_FR_BR_m1=float('inf')
    loss_FR_BL_BR_m1=loss_FR_BL_BR_m2=loss_FR_B1_BL_BR_m1=loss_FR_B1_BL_BR_m2=loss_FR_B1_B2_BL_BR_m1=loss_FR_B1_B2_BL_BR_m2=float('inf')
    loss_FL_B1_m1=loss_FL_B1_m2=loss_FL_B1_B2_m1=loss_FL_B1_B2_m2=float('inf')
    loss_FL_B1_BL_m1=loss_FL_B1_BL_m2=loss_FL_B1_BR_m1=loss_FL_B1_BR_m2=loss_FL_BL_m2=loss_FL_BL_m1=loss_FL_BR_m2=loss_FL_BR_m1=float('inf')
    loss_FL_BL_BR_m1=loss_FL_BL_BR_m2=loss_FL_B1_BL_BR_m1=loss_FL_B1_BL_BR_m2=loss_FL_B1_B2_BL_BR_m1=loss_FL_B1_B2_BL_BR_m2=float('inf')
    loss_F1_F2_B1_m1=loss_F1_F2_B1_m2=loss_F1_F2_B1_B2_m1=loss_F1_F2_B1_B2_m2=float('inf')
    loss_F1_F2_B1_BL_m1=loss_F1_F2_B1_BL_m2=loss_F1_F2_B1_BR_m1=loss_F1_F2_B1_BR_m2=loss_F1_F2_BL_m2=loss_F1_F2_BL_m1=loss_F1_F2_BR_m2=loss_F1_F2_BR_m1=float('inf')
    loss_F1_F2_BL_BR_m1=loss_F1_F2_BL_BR_m2=loss_F1_F2_B1_BL_BR_m1=loss_F1_F2_B1_BL_BR_m2=loss_F1_F2_B1_B2_BL_BR_m1=loss_F1_F2_B1_B2_BL_BR_m2=float('inf')
    loss_F1_F2_FL_BR_m1=loss_F1_F2_FL_BR_m2=loss_F1_F2_FL_B1_BR_m1=loss_F1_F2_FL_B1_BR_m2=loss_F1_F2_FL_FR_B1_BR_m1=loss_F1_F2_FL_FR_B1_BR_m2=loss_F1_F2_FL_FR_B1_B2_BR_m1=loss_F1_F2_FL_FR_B1_B2_BR_m2=float('inf')
    loss_F1_F2_FL_FR_B1_BL_BR_m1=loss_F1_F2_FL_FR_B1_BL_BR_m2=loss_F1_F2_FL_B1_B2_BL_m1=loss_F1_F2_FL_B1_B2_BL_m2=loss_F1_FL_B1_B2_BL_m1=loss_F1_FL_B1_B2_BL_m2=float('inf')
    vehicle_id=[]
    df = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\Features_classificationGP070239_lane_VideoNo.txt',delimiter ="\t").fillna(value = 0)
    df = df[df['ID'] < 45028]
    df_x = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\x.txt',delimiter ="\t").fillna(value = 0)
    df['x']=df_x
    df_y = pd.read_csv(r'C:\Users\Pooja\Desktop\2ndsem\HiWi\Jupyter_New_06.09\Jupyter_New_06.09\y.txt',delimiter ="\t").fillna(value = 0)
    df['y']=df_y 
    x = df['x'].as_matrix()
    y = df['y'].as_matrix()
    displacement_x=np.zeros((x.shape),dtype=np.float32)
    displacement_y=np.zeros((y.shape),dtype=np.float32)
    for i in range(1,x.shape[0]):
        displacement_x[i] = x[i]-x[i-1]
        displacement_y[i] = y[i]-y[i-1]
        displacement = np.sqrt(np.square(displacement_x)+np.square(displacement_y))
        new_speed = displacement/0.04
    df['speed']=new_speed
    df_ids = df[df.x < 15]
    le = LabelEncoder()
    le.fit(df_ids['ID'])
    vehicle_id = le.classes_
    val = np.array([537])
    vehicle_id=np.setdiff1d(vehicle_id,val)
    vehicle_id = vehicle_id[:int(len(vehicle_id) *0.02)]
    
    test_ids=vehicle_id[int(len(vehicle_id) *0.95):] #[126,124,5,15,31,51,32,70,32,119,19,108,80,79,33,121,29,113]   #vehicle_id[int(len(vehicle_id) *0.95):] #48 -FL  #v
    #test_ids = np.insert(test_ids, 0, 1)
    #vehicle_id = veh_id #[veh_id]
    vehicle_found = False
    #deep_maxent initialization check
    write_true = False
    list_vids= defaultdict(set)
    first_ego_m1=True
    first_ego_m2=True
    first_ego_FL_m1=True
    first_ego_FL_m2=True
    first_ego_FR_m1= True
    first_ego_FR_m2= True
    first_ego_F1_m1= True
    first_ego_F1_m2= True
    first_ego_F1_F2_m1= True
    first_ego_F1_F2_m2= True
    first_ego_F1_FR_m1= True
    first_ego_F1_FR_m2= True
    first_ego_F1_FL_m1= True
    first_ego_F1_FL_m2= True
    first_ego_FL_FR_m1= True
    first_ego_FL_FR_m2= True
    first_ego_F1_FR_F2_m1= True
    first_ego_F1_FR_F2_m2= True
    first_ego_F1_FL_F2_m1= True
    first_ego_F1_FL_F2_m2= True
    first_ego_F1_FR_FL_m1= True
    first_ego_F1_FR_FL_m2= True
    first_ego_F1_F2_FR_FL_m1= True
    first_ego_F1_F2_FR_FL_m2= True
    first_ego_F1_F2_FR_BR_m1= True
    first_ego_F1_F2_FR_BR_m2= True
    first_ego_F1_F2_FR_B1_BR_m1= True
    first_ego_F1_F2_FR_B1_BR_m2= True
    first_ego_F1_FR_B1_BR_m1= True
    first_ego_F1_FR_B1_BR_m2= True
    ####################################
    first_ego_BL_m1=True
    first_ego_BL_m2=True
    first_ego_BR_m1= True
    first_ego_BR_m2= True
    first_ego_B1_m1= True
    first_ego_B1_m2= True
    first_ego_B1_B2_m1= True
    first_ego_B1_B2_m2= True
    first_ego_B1_BR_m1= True
    first_ego_B1_BR_m2= True
    first_ego_B1_BL_m1= True
    first_ego_B1_BL_m2= True
    first_ego_BL_BR_m1= True
    first_ego_BL_BR_m2= True
    first_ego_B1_BR_B2_m1= True
    first_ego_B1_BR_B2_m2= True
    first_ego_B1_BL_B2_m1= True
    first_ego_B1_BL_B2_m2= True
    first_ego_B1_BR_BL_m1= True
    first_ego_B1_BR_BL_m2= True
    first_ego_B1_B2_BR_BL_m1= True
    first_ego_B1_B2_BR_BL_m2= True
    ######################################
    first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m1 = True
    first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m2= True
    first_ego_F1_B1_m1= True
    first_ego_F1_B1_m2 = True
    first_ego_F1_BL_m1= True
    first_ego_F1_BL_m2 = True
    first_ego_F1_BR_m1= True
    first_ego_F1_BR_m2 = True
    first_ego_F1_B1_B2_m1= True
    first_ego_F1_B1_B2_m2 = True
    first_ego_F1_B1_BL_m1= True
    first_ego_F1_B1_BL_m2 = True
    first_ego_F1_B1_BR_m1= True
    first_ego_F1_B1_BR_m2 = True
    first_ego_F1_BL_BR_m1= True
    first_ego_F1_BL_BR_m2 = True
    first_ego_F1_B1_BL_BR_m1= True
    first_ego_F1_B1_BL_BR_m2 = True
    first_ego_F1_B1_B2_BL_BR_m1= True
    first_ego_F1_B1_B2_BL_BR_m2 = True
    first_ego_F1_B1_B2_BL_m1= True
    first_ego_F1_B1_B2_BL_m2 = True
    first_ego_FL_B1_m1= True
    first_ego_FL_B1_m2 = True
    first_ego_FL_BL_m1= True
    first_ego_FL_BL_m2 = True
    first_ego_FL_BR_m1= True
    first_ego_FL_BR_m2 = True
    first_ego_FL_B1_B2_m1= True
    first_ego_FL_B1_B2_m2 = True
    first_ego_FL_B1_BL_m1= True
    first_ego_FL_B1_BL_m2 = True
    first_ego_FL_B1_BR_m1= True
    first_ego_FL_B1_BR_m2 = True
    first_ego_FL_BL_BR_m1= True
    first_ego_FL_BL_BR_m2 = True
    first_ego_FL_B1_BL_BR_m1= True
    first_ego_FL_B1_BL_BR_m2 = True
    first_ego_FL_B1_B2_BL_BR_m1= True
    first_ego_FL_B1_B2_BL_BR_m2 = True
    first_ego_FL_B1_B2_BL_m1= True
    first_ego_FL_B1_B2_BL_m2 = True
    first_ego_FR_B1_m1= True
    first_ego_FR_B1_m2 = True
    first_ego_FR_BL_m1= True
    first_ego_FR_BL_m2 = True
    first_ego_FR_BR_m1= True
    first_ego_FR_BR_m2 = True
    first_ego_FR_B1_B2_m1= True
    first_ego_FR_B1_B2_m2 = True
    first_ego_FR_B1_BL_m1= True
    first_ego_FR_B1_BL_m2 = True
    first_ego_FR_B1_BR_m1= True
    first_ego_FR_B1_BR_m2 = True
    first_ego_FR_BL_BR_m1= True
    first_ego_FR_BL_BR_m2 = True
    first_ego_FR_B1_BL_BR_m1= True
    first_ego_FR_B1_BL_BR_m2 = True
    first_ego_FR_B1_B2_BL_BR_m1= True
    first_ego_FR_B1_B2_BL_BR_m2 = True
#########################################################################################################
    first_ego_F1_F2_B1_m1= True
    first_ego_F1_F2_B1_m2 = True
    first_ego_F1_F2_BL_m1= True
    first_ego_F1_F2_BL_m2 = True
    first_ego_F1_F2_BR_m1= True
    first_ego_F1_F2_BR_m2 = True
    first_ego_F1_F2_B1_B2_m1= True
    first_ego_F1_F2_B1_B2_m2 = True
    first_ego_F1_F2_B1_BL_m1= True
    first_ego_F1_F2_B1_BL_m2 = True
    first_ego_F1_F2_B1_BR_m1= True
    first_ego_F1_F2_B1_BR_m2 = True
    first_ego_F1_F2_BL_BR_m1= True
    first_ego_F1_F2_BL_BR_m2 = True
    first_ego_F1_F2_B1_BL_BR_m1= True
    first_ego_F1_F2_B1_BL_BR_m2 = True
    first_ego_F1_F2_B1_B2_BL_BR_m1= True
    first_ego_F1_F2_B1_B2_BL_BR_m2 = True
    first_ego_F1_F2_B1_B2_BL_m1= True
    first_ego_F1_F2_B1_B2_BL_m2 = True
    #######################################################################################################
    first_ego_F1_F2_FL_BR_m1 = True
    first_ego_F1_F2_FL_BR_m2 = True
    first_ego_F1_F2_FL_BL_m1 = True
    first_ego_F1_F2_FL_BL_m2 = True

    first_ego_F1_F2_FL_BL_BR_m1 = True
    first_ego_F1_F2_FL_BL_BR_m2 = True
    first_ego_F1_F2_FL_FR_BL_BR_m1 = True
    first_ego_F1_F2_FL_FR_BL_BR_m2 = True
    first_ego_FL_FR_B1_B2_BL_BR_m1 = True
    first_ego_FL_FR_B1_B2_BL_BR_m2 = True
    first_ego_FL_FR_B1_B2_BL_m1 = True
    first_ego_FL_FR_B1_B2_BL_m2 = True

    first_ego_F1_F2_FL_B1_BL_m1 = True
    first_ego_F1_F2_FL_B1_BL_m2 = True
    first_ego_F1_F2_FL_B1_BL_BR_m1 = True
    first_ego_F1_F2_FL_B1_BL_BR_m2 = True
    first_ego_F1_FL_B1_BL_m1 = True
    first_ego_F1_FL_B1_BL_m2 = True
    first_ego_F1_F2_FL_B1_BR_m1 = True
    first_ego_F1_F2_FL_B1_BR_m2 = True
    first_ego_F1_F2_FL_B1_m1 = True
    first_ego_F1_F2_FL_B1_m2 = True
    first_ego_F1_F2_FL_FR_B1_BR_m1= True
    first_ego_F1_F2_FL_FR_B1_BR_m2= True
    first_ego_F1_F2_FL_FR_B1_BL_m1= True
    first_ego_F1_F2_FL_FR_B1_BL_m2= True
    first_ego_F1_FL_FR_B1_BL_m1= True
    first_ego_F1_FL_FR_B1_BL_m2= True
    first_ego_F1_FL_FR_B1_B2_BL_m1= True
    first_ego_F1_FL_FR_B1_B2_BL_m2= True
    first_ego_F1_F2_FL_FR_BR_m1= True
    first_ego_F1_F2_FL_FR_BR_m2= True
    first_ego_F1_F2_FL_FR_B1_B2_BR_m1=True
    first_ego_F1_F2_FL_FR_B1_B2_BR_m2=True
    first_ego_F1_F2_FL_FR_B1_BL_BR_m1=True
    first_ego_F1_F2_FL_FR_B1_BL_BR_m2=True
    first_ego_F1_F2_FL_B1_B2_BL_m1= True
    first_ego_F1_F2_FL_B1_B2_BL_m2=True
    first_ego_F1_FL_B1_B2_BL_m1=True
    first_ego_F1_FL_B1_B2_BL_m2=True
    first_ego_F1_F2_FR_B1_BL_BR_m1=True
    first_ego_F1_F2_FR_B1_BL_BR_m2=True
    first_ego_F1_F2_FR_B1_BL_m1=True
    first_ego_F1_F2_FR_B1_BL_m2=True
    first_ego_F1_FR_B1_BL_BR_m1=True
    first_ego_F1_FR_B1_BL_BR_m2=True
    first_ego_F1_FR_B1_B2_BL_BR_m1=True
    first_ego_F1_FR_B1_B2_BL_BR_m2=True
################################################################################################################
    first_ego_F1_FL_FR_B1_B2_BL_BR_m1=True
    first_ego_F1_FL_FR_B1_B2_BL_BR_m2=True
    first_ego_F1_FL_FR_B1_BR_m1=True
    first_ego_F1_FL_FR_B1_BR_m2=True
    first_ego_F1_FL_FR_B1_B2_BR_m1=True
    first_ego_F1_FL_FR_B1_B2_BR_m2=True
    first_ego_F1_FL_FR_B1_B2_m1=True
    first_ego_F1_FL_FR_B1_B2_m2=True
    first_ego_F1_FR_B1_B2_m1=True
    first_ego_F1_FR_B1_B2_m2=True
    first_ego_F1_FR_B1_BL_m1=True
    first_ego_F1_FR_B1_BL_m2=True
    first_ego_FR_B1_B2_BL_m1=True
    first_ego_FR_B1_B2_BL_m2=True


    #gridowrld for each situation
    gw_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_FR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_FR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_F2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_F2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_F2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_F2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FR_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FR_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)

    gw_F1_FR_FL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_FL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FR_FL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_FL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    ###############################################################################################
    gw_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_BR_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_BR_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_BL_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_BL_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_BR_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_BR_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_B1_B2_BR_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_B1_B2_BR_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    ##############################################################################################
    gw_F1_F2_FR_FL_B1_B2_BR_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_FL_B1_B2_BR_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_B2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_B2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_B2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_B2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)

    gw_FR_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_B2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_B2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)

    gw_FL_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_B2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_B2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_B1_B2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_B1_B2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
####################################################################################################
    gw_F1_F2_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_B2_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_B2_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_B2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_B2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_B1_B2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_B1_B2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    ##############################################################################################################
    gw_F1_F2_FL_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_FR_B1_B2_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_FR_B1_B2_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FL_FR_B1_B2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FL_FR_B1_B2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_B1_BL_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_B1_BL_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_B1_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_B1_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_B1_B2_BL_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_B1_B2_BL_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_B1_BR_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_B1_BR_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_B1_m1 = gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_B1_m2 = gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_B1_BR_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_B1_BR_m2= gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_B1_BL_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_B1_BL_m2= gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_BL_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_BL_m2= gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_B2_BL_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_B2_BL_m2= gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_BR_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_BR_m2= gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_B1_B2_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_B1_B2_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_FR_B1_BL_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_FR_B1_BL_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FL_B1_B2_BL_m1= gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FL_B1_B2_BL_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_B1_B2_BL_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_B1_B2_BL_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_B2_BL_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_B2_BL_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_B2_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_B2_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FL_FR_B1_B2_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FL_FR_B1_B2_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_B1_B2_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_B1_B2_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_B1_BL_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_B1_BL_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_FR_B1_B2_BL_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_FR_B1_B2_BL_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FR_B1_BL_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_B1_BL_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_F2_FR_B1_BL_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_F2_FR_B1_BL_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_B1_BL_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_B1_BL_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    gw_F1_FR_B1_B2_BL_BR_m1=gridworld.Gridworld(grid_size_x,grid_size_y, wind, discount,border_m1)
    gw_F1_FR_B1_B2_BL_BR_m2=gridworld.Gridworld(14,20, wind, discount,border_m2)
    ###########################################################################################################
    #NN for each situation
    deep_max_m1=deep_maxent.Deep_Maxent()
    deep_max_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_FR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_FR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_F2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_F2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_F2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_F2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_FL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_FL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_FL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_FL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
    ##############################################################################################
    deep_max_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_BL_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_BL_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_BR_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_B1_B2_BR_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_B1_B2_BR_BL_m2=deep_maxent.Deep_Maxent()
    ############################################################################################
    deep_max_F1_F2_FR_FL_B1_B2_BR_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FR_FL_B1_B2_BR_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_B1_B2_BL_m2=deep_maxent.Deep_Maxent()

    deep_max_FL_B1_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FL_B1_B2_BL_m2=deep_maxent.Deep_Maxent()

    deep_max_FR_B1_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
##################################################################################################
    deep_max_F1_F2_B1_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_BL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_BL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_FL_FR_B1_B2_BL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_FL_FR_B1_B2_BL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_FL_FR_B1_B2_BL_m1 = deep_maxent.Deep_Maxent()
    deep_max_FL_FR_B1_B2_BL_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_B1_B2_BL_m2=deep_maxent.Deep_Maxent()
#########################################################################################################
    deep_max_F1_F2_FL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BL_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BL_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BL_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BL_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_FL_B1_BL_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_FL_B1_BL_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BR_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_BR_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_m1 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_m2 = deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BR_m1= deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BR_m2= deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BL_m1= deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BL_m2= deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_BL_m1= deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_BL_m2= deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BL_m1= deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BL_m2= deep_maxent.Deep_Maxent()

    deep_max_F1_F2_FL_FR_B1_B2_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_B2_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_B1_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_B2_BL_m1= deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_B1_B2_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_B1_B2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_B1_B2_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BL_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BL_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_BR_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FL_FR_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_B2_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_B2_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_FR_B1_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_BL_m1=deep_maxent.Deep_Maxent()
    deep_max_FR_B1_B2_BL_m2=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_BR_m1=deep_maxent.Deep_Maxent()
    deep_max_F1_F2_FL_FR_BR_m2=deep_maxent.Deep_Maxent()
############################################################################################################
    xmin = 0
    ymin = 0
    #vehicle_id = np.delete(vehicle_id,0) #remove vehicle 1
    
    #reward for each situation
    avg_reward_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_m1 = np.zeros((gw_m1.n_states))
    avg_reward_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_FL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_FR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_F2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_FR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_FL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_FL_FR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_FR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_FR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_FR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_FR_F2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_F2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_F2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_F2_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_F1_FL_F2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_F2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_F2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_F2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_FL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_FL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_FL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_FL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_FL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_FL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_FL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_FL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_BL_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_BR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_B2_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_BR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_BL_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_BL_BR_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_BR_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_BR_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_BR_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_BR_B2_m2 = np.zeros((gw_m2.n_states))
    
    avg_reward_B1_BL_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_BL_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_BL_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_BL_B2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_B1_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_BR_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_BR_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_B1_B2_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_B1_B2_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_B1_B2_BR_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_B1_B2_BR_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m2 = np.zeros((gw_m2.n_states))
######################################################################################
    avg_reward_F1_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_B2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_B2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_B2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
########################################################################
    avg_reward_F1_F2_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_B2_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_FL_FR_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_FR_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_FL_FR_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_FR_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
##########################################################################################3
    avg_reward_F1_F2_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_B1_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_B1_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_B1_B2_BL_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_B1_B2_BL_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_B1_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_BR_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_BR_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_BR_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_B1_m1 = np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_B1_m2 = np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_m1 = np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_m2 = np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_B1_BR_m1= np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_B1_BR_m2= np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m1= np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m2= np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_B1_BL_m1= np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_B1_BL_m2= np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m1= np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m2= np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_BL_m1= np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_BL_m2= np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_BL_m1= np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_BL_m2= np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_B2_BL_m1= np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_B2_BL_m2= np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m1= np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m2= np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_B1_B2_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_B1_B2_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_B1_B2_BL_m1= np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_B1_B2_BL_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m1= np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_B1_B2_BL_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_B1_B2_BL_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_B1_B2_BL_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_B1_B2_BL_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_B2_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_B2_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_B2_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_B2_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_B2_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FR_B1_BL_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FR_B1_BL_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BL_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FR_B1_BL_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_B1_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_B1_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_FL_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_FL_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_B1_B2_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_B1_B2_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_B1_B2_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_B1_B2_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FR_B1_BL_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FR_B1_BL_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FR_B1_BL_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FR_B1_BL_m2=np.zeros((gw_m2.n_states))

    avg_reward_FR_B1_B2_BL_m1=np.zeros((gw_m1.n_states))
    avg_reward_FR_B1_B2_BL_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_FR_B1_B2_BL_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_FR_B1_B2_BL_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_FL_FR_B1_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_FL_FR_B1_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_FL_FR_B1_BR_m2=np.zeros((gw_m2.n_states))

    avg_reward_F1_F2_FL_FR_BR_m1=np.zeros((gw_m1.n_states))
    avg_reward_F1_F2_FL_FR_BR_m2=np.zeros((gw_m2.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_BR_m1=np.zeros((gw_m1.n_states))
    avg_grnd_rwd_F1_F2_FL_FR_BR_m2=np.zeros((gw_m2.n_states))

    i=0
    last_frame = 100#int(df['Time'].max())
    print('last frame:',last_frame)
    for time in range(0,last_frame,120):
        if (time%10000==0):
            print('Current time:',time)
        for v_id in vehicle_id:
            #print('Vehicle ID:',v_id)
            df_in=df.loc[df['ID']==v_id]
            if (df_in['Type'].iloc[0]== 'Car'):
                trajectories=[]
                obs_trajectories=[]
                current_states=[]
                if time in df_in['Time'].values:
                    ID_F=df_in.loc[( (df_in['Time']==time)), 'ID_F'].iloc[0]
                    ID_F2=df_in.loc[( (df_in['Time']==time)), 'ID_F2'].iloc[0]
                    ID_B=df_in.loc[((df_in['Time']==time)), 'ID_B'].iloc[0]
                    ID_B2=df_in.loc[( (df_in['Time']==time)), 'ID_B2'].iloc[0]
                    ID_FL=df_in.loc[( (df_in['Time']==time)), 'ID_FL'].iloc[0]
                    ID_BL=df_in.loc[( (df_in['Time']==time)), 'ID_BL'].iloc[0]
                    ID_BR=df_in.loc[( (df_in['Time']==time)), 'ID_BR'].iloc[0]
                    ID_FR=df_in.loc[((df_in['Time']==time)), 'ID_FR'].iloc[0]
                    x = df_in.loc[( (df_in['Time']==time)), 'x'].iloc[0]
                    y = df_in.loc[( (df_in['Time']==time)), 'y'].iloc[0]
                    #print(ID_F,ID_F2, ID_B, ID_B2,ID_FL,ID_BL,ID_BR, ID_FR)
                    if ((ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR_FL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_FL_m1
                        else:
                            gw = gw_F1_FR_FL_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        #print(obs_trajectories,F2_x,F2_y,F2_n_x,F2_n_y,determine_action(F2_x,F2_y,F2_n_x,F2_n_y))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_FR and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FR_FL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_FL_m1
                        else:
                            gw = gw_F1_F2_FR_FL_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        #print(obs_trajectories,F2_x,F2_y,F2_n_x,F2_n_y,determine_action(F2_x,F2_y,F2_n_x,F2_n_y))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2) and not (ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR_F2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_F2_m1
                        else:
                            gw = gw_F1_FL_F2_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y))))) 
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        #print(obs_trajectories,F2_x,F2_y,F2_n_x,F2_n_y,determine_action(F2_x,F2_y,F2_n_x,F2_n_y))
                        vehicle_found = True
                    elif ((ID_F and ID_FR and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR_F2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_F2_m1
                        else:
                            gw = gw_F1_FR_F2_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y))))) 
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y))))) 
                        vehicle_found = True
                    if ((ID_F and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                        add_vids_to_list(list_vids,'F1_F2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_m1
                        else:
                            gw = gw_F1_F2_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y))))) 
                        vehicle_found = True
                    elif ((ID_F and ID_FR) and not (ID_F2 or ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_m1
                        else:
                            gw = gw_F1_FR_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y))))) 
                        vehicle_found = True
                    elif ((ID_FL and ID_FR) and not (ID_F2 or ID_F or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'FL_FR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        if (x<8 and y<25):
                            gw = gw_FL_FR_m1
                        else:
                            gw = gw_FL_FR_m2
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y))))) 
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_F2 and ID_B and ID_BL and ID_BR) and not(ID_FL or ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_FR_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_B1_BL_BR_m1
                        else:
                            gw = gw_F1_F2_FR_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_F2 and ID_B and ID_BL) and not(ID_FL or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FR_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_B1_BL_m1
                        else:
                            gw = gw_F1_F2_FR_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
##########
                    elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR) and not(ID_FL or ID_B2 or ID_F2)):
                        add_vids_to_list(list_vids,'F1_FR_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_BL_BR_m1
                        else:
                            gw = gw_F1_FR_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not(ID_FL or ID_F2)):
                        add_vids_to_list(list_vids,'F1_FR_B1_B2_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        print('ID_BL:',ID_BL,time)
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_B2_BL_BR_m1
                        else:
                            gw = gw_F1_FR_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_FL and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not(ID_F or ID_F2)):
                        add_vids_to_list(list_vids,'FL_FR_B1_B2_BL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FL_FR_B1_B2_BL_BR_m1
                        else:
                            gw = gw_FL_FR_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True


                    elif ((ID_F and ID_FL) and not (ID_F2 or ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_m1
                        else:
                            gw = gw_F1_FL_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y))))) 
                        vehicle_found = True
                    elif (ID_FL and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                        add_vids_to_list(list_vids,'FL',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        if (x<8 and y<25):
                            gw = gw_FL_m1
                        else:
                            gw = gw_FL_m2
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))  
                        vehicle_found = True
                    
                    elif (ID_F and not (ID_FL or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                        add_vids_to_list(list_vids,'F1',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        if (x<8 and y<25):
                            gw = gw_F1_m1
                        else:
                            gw = gw_F1_m2
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))  
                        vehicle_found = True
                
                    elif (ID_FR and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                        #print('ID:',v_id)
                        #print(ID_FR,ID_F2, ID_B, ID_B2,ID_FL,ID_BL,ID_BR, ID_F)
                        #print('Type:',df_in['Type'].iloc[0])
                        #print('time:',time)
                        #print(df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'])
                        add_vids_to_list(list_vids,'FR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        if (x<8 and y<25):
                            gw = gw_FR_m1
                        else:
                            gw = gw_FR_m2
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))  
                        vehicle_found = True

#################################################################################################################
                    elif ((ID_B and ID_BL and ID_BR) and not (ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_BR_BL',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_B1_BR_BL_m1
                        else:
                            gw = gw_B1_BR_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        #print(obs_trajectories,B2_x,B2_y,B2_n_x,B2_n_y,determine_action(B2_x,B2_y,B2_n_x,B2_n_y))
                        vehicle_found = True
                    elif ((ID_B and ID_BL and ID_BR and ID_B2) and not (ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_B2_BR_BL',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_B1_B2_BR_BL_m1
                        else:
                            gw = gw_B1_B2_BR_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        #print(obs_trajectories,B2_x,B2_y,B2_n_x,B2_n_y,determine_action(B2_x,B2_y,B2_n_x,B2_n_y))
                        vehicle_found = True
                    elif ((ID_B and ID_BL and ID_B2) and not (ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_BL_B2',v_id)
                        print('ID_BL:',ID_BL,time)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_B1_BL_B2_m1
                        else:
                            gw = gw_B1_BL_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y))))) 
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        #print(obs_trajectories,B2_x,B2_y,B2_n_x,B2_n_y,determine_action(B2_x,B2_y,B2_n_x,B2_n_y))
                        vehicle_found = True
                    elif ((ID_B and ID_BR and ID_B2) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_BR_B2',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_B1_BR_B2_m1
                        else:
                            gw = gw_B1_BR_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y))))) 
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y))))) 
                        vehicle_found = True
                    elif ((ID_B and ID_B2) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                        add_vids_to_list(list_vids,'B1_B2',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_B1_B2_m1
                        else:
                            gw = gw_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y))))) 
                        vehicle_found = True
                    elif ((ID_B and ID_BR) and not (ID_B2 or ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_BR',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_B1_BR_m1
                        else:
                            gw = gw_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y))))) 
                        vehicle_found = True
                    elif ((ID_BL and ID_BR) and not (ID_B2 or ID_B or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'BL_BR',v_id)
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_BL_BR_m1
                        else:
                            gw = gw_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y))))) 
                        vehicle_found = True
                    elif ((ID_B and ID_BL) and not (ID_B2 or ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                        add_vids_to_list(list_vids,'B1_BL',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_B1_BL_m1
                        else:
                            gw = gw_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y))))) 
                        vehicle_found = True
                    elif (ID_BL and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                        add_vids_to_list(list_vids,'BL',v_id)
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]  
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_BL_m1
                        else:
                            gw = gw_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))  
                        vehicle_found = True
                    
                    elif (ID_B and not (ID_BL or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                        add_vids_to_list(list_vids,'B1',v_id)
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_B1_m1
                        else:
                            gw = gw_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))  
                        vehicle_found = True
                
                    elif (ID_BR and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                        #print('ID:',v_id)
                        #print(ID_BR,ID_B2, ID_B, ID_B2,ID_BL,ID_BL,ID_BR, ID_B)
                        #print('Type:',df_in['Type'].iloc[0])
                        #print('time:',time)
                        #print(df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'])
                        add_vids_to_list(list_vids,'BR',v_id)
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_BR_m1
                        else:
                            gw = gw_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y)))))  
                        vehicle_found = True
                    
                    elif not (ID_F or ID_F2 or ID_B or ID_B2 or ID_FL or ID_BL or ID_BR or ID_FR):
                        add_vids_to_list(list_vids,'no_obs',v_id)
                        #print('ID no obstacle:',ID_FL,ID_F,ID_F2, ID_B, ID_B2,ID_BL,ID_BR, ID_FR)
                        if (x<8 and y<25):
                            gw = gw_m1
                        else:
                            gw = gw_m2
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_F2 and ID_B and ID_B2 and ID_BL and ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FR_FL_B1_B2_BR_BL',v_id)
                        print('ID_B:',v_id,ID_B,time)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_FL_B1_B2_BR_BL_m1
                        else:
                            gw = gw_F1_F2_FR_FL_B1_B2_BR_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_F2 and ID_BR) and not(ID_FL or ID_B or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FR_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_BR_m1
                        else:
                            gw = gw_F1_F2_FR_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_F2 and ID_BR and ID_B) and not(ID_FL or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FR_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FR_B1_BR_m1
                        else:
                            gw = gw_F1_F2_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                                determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                                int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_BR and ID_B) and not(ID_FL or ID_B2 or ID_BL or ID_F2)):
                        add_vids_to_list(list_vids,'F1_F2_FR_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_BR_m1
                        else:
                            gw = gw_F1_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                                determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                                int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_B and ID_B2 and ID_BL and ID_BR) and not(ID_F2)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_B2_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_B2_BL_BR_m1
                        else:
                            gw = gw_F1_FL_FR_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_B and ID_B2) and not(ID_F2 or ID_FL or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR_B1_B2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_B2_m1
                        else:
                            gw = gw_F1_FR_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_B and ID_BL) and not(ID_F2 or ID_FL or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FR_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_BL_m1
                        else:
                            gw = gw_F1_FR_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_FR and ID_B and ID_B2 and ID_BL) and not(ID_F or ID_F2 or ID_FL or ID_BR)):
                        add_vids_to_list(list_vids,'FR_B1_B2_BL',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_B2_BL_m1
                        else:
                            gw = gw_FR_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_B and ID_B2 and ID_BR) and not(ID_F2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_B2_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_B2_BR_m1
                        else:
                            gw = gw_F1_FL_FR_B1_B2_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_B and ID_B2) and not(ID_F2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_B2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_B2_m1
                        else:
                            gw = gw_F1_FL_FR_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_B and ID_BR) and not(ID_F2 or ID_B2 or ID_BL )):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_BR_m1
                        else:
                            gw = gw_F1_FL_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_FR and ID_B and ID_B2) and not(ID_F2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_B2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_B2_m1
                        else:
                            gw = gw_F1_FL_FR_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FR and ID_B and ID_BR) and not(ID_F2 or ID_B2 or ID_BL  or ID_FL)):
                        add_vids_to_list(list_vids,'F1_FR_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_FR_B1_BR_m1
                        else:
                            gw = gw_F1_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
###############################################################################################################
                    elif ((ID_F and ID_BR and ID_F2) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_BL or ID_B)):
                        add_vids_to_list(list_vids,'F1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_BR_m1
                        else:
                            gw = gw_F1_F2_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_BL and ID_F2) and not (ID_FR or ID_FL or ID_B2 or ID_BR or ID_B)):
                        add_vids_to_list(list_vids,'F1_F2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_BL_m1
                        else:
                            gw = gw_F1_F2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_F2) and not (ID_FR or ID_FL or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_m1
                        else:
                            gw = gw_F1_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2 and ID_F2) and not (ID_FR or ID_FL or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_B1_B2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_B2_m1
                        else:
                            gw = gw_F1_F2_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x, B2_y))),
                                                 determine_action(B2_x, B2_y, B2_n_x, B2_n_y),
                                                 int(gw.point_to_int((B2_n_x, B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BL and ID_F2) and not (ID_FR or ID_FL or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_BL_m1
                        else:
                            gw = gw_F1_F2_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x, BL_y))),
                                                 determine_action(BL_x, BL_y, BL_n_x, BL_n_y),
                                                 int(gw.point_to_int((BL_n_x, BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BR and ID_F2) and not (ID_FR or ID_FL or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_BR_m1
                        else:
                            gw = gw_F1_F2_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_BL and ID_BR and ID_F2) and not (ID_FR or ID_FL or ID_B2 or ID_B)):
                        add_vids_to_list(list_vids,'F1_F2_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_BL_BR_m1
                        else:
                            gw = gw_F1_F2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BL and ID_BR and ID_F2) and not (ID_FR or ID_FL or ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_BL_BR_m1
                        else:
                            gw = gw_F1_F2_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_BR and ID_F2) and not (ID_FR or ID_FL)):
                        add_vids_to_list(list_vids,'F1_F2_B1_B2_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_B2_BL_BR_m1
                        else:
                            gw = gw_F1_F2_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_F2) and not (ID_FR or ID_FL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_B1_B2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_B1_B2_BL_m1
                        else:
                            gw = gw_F1_F2_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
#################################################################################################################
                    elif ((ID_F and ID_BR) and not (ID_FR or ID_FL or ID_B2 or ID_BL or ID_B)):
                        add_vids_to_list(list_vids,'F1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_BR_m1
                        else:
                            gw = gw_F1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_BL) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_BR or ID_B)):
                        add_vids_to_list(list_vids,'F1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_BL_m1
                        else:
                            gw = gw_F1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_m1
                        else:
                            gw = gw_F1_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2) and not (ID_FR or ID_FL or ID_F2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_B1_B2',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_B2_m1
                        else:
                            gw = gw_F1_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x, B2_y))),
                                                 determine_action(B2_x, B2_y, B2_n_x, B2_n_y),
                                                 int(gw.point_to_int((B2_n_x, B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BL) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_BL_m1
                        else:
                            gw = gw_F1_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x, BL_y))),
                                                 determine_action(BL_x, BL_y, BL_n_x, BL_n_y),
                                                 int(gw.point_to_int((BL_n_x, BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BR) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_BR_m1
                        else:
                            gw = gw_F1_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_BL and ID_BR) and not (ID_FR or ID_FL or ID_F2 or ID_B2 or ID_B)):
                        add_vids_to_list(list_vids,'F1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_BL_BR_m1
                        else:
                            gw = gw_F1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_B and ID_BL and ID_BR) and not (ID_FR or ID_FL or ID_F2 or ID_B2)):
                        add_vids_to_list(list_vids,'F1_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_BL_BR_m1
                        else:
                            gw = gw_F1_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2 and ID_BL and ID_BR) and not (ID_F2 or ID_FR or ID_FL)):
                        add_vids_to_list(list_vids,'F1_B1_B2_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_B2_BL_BR_m1
                        else:
                            gw = gw_F1_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_B and ID_B2 and ID_BL) and not (ID_F2 or ID_FR or ID_FL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_B1_B2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_B1_B2_BL_m1
                        else:
                            gw = gw_F1_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        vehicle_found = True
##############################################################################################################FR combinatio
                    elif ((ID_FR and ID_BR) and not (ID_F or ID_FL or ID_B2 or ID_BL or ID_B)):
                        add_vids_to_list(list_vids,'FR_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FR_BR_m1
                        else:
                            gw = gw_FR_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_BL) and not (ID_F or ID_FL or ID_F2 or ID_B2 or ID_BR or ID_B)):
                        add_vids_to_list(list_vids,'FR_BL',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FR_BL_m1
                        else:
                            gw = gw_FR_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_B) and not (ID_F or ID_FL or ID_F2 or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'FR_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_m1
                        else:
                            gw = gw_FR_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_FR and ID_B and ID_B2) and not (ID_F or ID_FL or ID_F2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'FR_B1_B2',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_B2_m1
                        else:
                            gw = gw_FR_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x, B2_y))),
                                                 determine_action(B2_x, B2_y, B2_n_x, B2_n_y),
                                                 int(gw.point_to_int((B2_n_x, B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_B and ID_BL) and not (ID_F or ID_FL or ID_F2 or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'FR_B1_BL',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_BL_m1
                        else:
                            gw = gw_FR_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x, BL_y))),
                                                 determine_action(BL_x, BL_y, BL_n_x, BL_n_y),
                                                 int(gw.point_to_int((BL_n_x, BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_B and ID_BR) and not (ID_F or ID_FL or ID_F2 or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'FR_B1_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_BR_m1
                        else:
                            gw = gw_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_BL and ID_BR) and not (ID_F or ID_FL or ID_F2 or ID_B2 or ID_B)):
                        add_vids_to_list(list_vids,'FR_BL_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FR_BL_BR_m1
                        else:
                            gw = gw_FR_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_B and ID_BL and ID_BR) and not (ID_F or ID_FL or ID_F2 or ID_B2)):
                        add_vids_to_list(list_vids,'FR_B1_BL_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_BL_BR_m1
                        else:
                            gw = gw_FR_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_FR and ID_F2 and ID_B and ID_B2 and ID_BL and ID_BR) and not (ID_F or ID_FL)):
                        add_vids_to_list(list_vids,'FR_B1_B2_BL_BR',v_id)
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FR_B1_B2_BL_BR_m1
                        else:
                            gw = gw_FR_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
###########################################################################################################FL combination
                    elif ((ID_FL and ID_BR) and not (ID_F or ID_FR or ID_B2 or ID_BL or ID_B)):
                        add_vids_to_list(list_vids,'FL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FL_BR_m1
                        else:
                            gw = gw_FL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                         determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                         int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_BL) and not (ID_F or ID_FR or ID_F2 or ID_B2 or ID_BR or ID_B)):
                        add_vids_to_list(list_vids,'FL_BL',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FL_BL_m1
                        else:
                            gw = gw_FL_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_B) and not (ID_F or ID_FR or ID_F2 or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'FL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_m1
                        else:
                            gw = gw_FL_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_FL and ID_B and ID_B2) and not (ID_F or ID_FR or ID_F2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'FL_B1_B2',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_B2_m1
                        else:
                            gw = gw_FL_B1_B2_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x, B2_y))),
                                                 determine_action(B2_x, B2_y, B2_n_x, B2_n_y),
                                                 int(gw.point_to_int((B2_n_x, B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_B and ID_BL) and not (ID_F or ID_FR or ID_F2 or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'FL_B1_BL',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_BL_m1
                        else:
                            gw = gw_FL_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x, BL_y))),
                                                 determine_action(BL_x, BL_y, BL_n_x, BL_n_y),
                                                 int(gw.point_to_int((BL_n_x, BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_B and ID_BR) and not (ID_F or ID_FR or ID_F2 or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'FL_B1_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_BR_m1
                        else:
                            gw = gw_FL_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_BL and ID_BR) and not (ID_F or ID_FR or ID_F2 or ID_B2 or ID_B)):
                        add_vids_to_list(list_vids,'FL_BL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FL_BL_BR_m1
                        else:
                            gw = gw_FL_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_B and ID_BL and ID_BR) and not (ID_F or ID_FR or ID_F2 or ID_B2)):
                        add_vids_to_list(list_vids,'FL_B1_BL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_BL_BR_m1
                        else:
                            gw = gw_FL_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_FL and ID_B and ID_B2 and ID_BL and ID_BR) and not (ID_F or ID_F2 or ID_FR)):
                        add_vids_to_list(list_vids,'FL_B1_B2_BL_BR',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_B2_BL_BR_m1
                        else:
                            gw = gw_FL_B1_B2_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x, BR_y))),
                                                 determine_action(BR_x, BR_y, BR_n_x, BR_n_y),
                                                 int(gw.point_to_int((BR_n_x, BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_FL and ID_B and ID_B2 and ID_BL) and not (ID_F or ID_F2 or ID_FR or ID_BR)):
                        add_vids_to_list(list_vids,'FL_B1_B2_BL',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_FL_B1_B2_BL_m1
                        else:
                            gw = gw_FL_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                         determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                         int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
##############################################################################################################
                    elif ((ID_F and ID_FL and ID_F2 and ID_BR) and not(ID_FR or ID_B or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_BR_m1
                        else:
                            gw = gw_F1_F2_FL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_BL) and not(ID_FR or ID_B or ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_FL_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_BL_BR_m1
                        else:
                            gw = gw_F1_F2_FL_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_BL and ID_FR) and not(ID_B or ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_FR_FL_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_BL_BR_m1
                        else:
                            gw = gw_F1_F2_FL_FR_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_B and ID_B2 and ID_FL and ID_BL and ID_FR) and not(ID_F or ID_F2 and ID_BR)):
                        add_vids_to_list(list_vids,'FL_FR_B1_B2_BL',v_id)
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_FL_FR_B1_B2_BL_m1
                        else:
                            gw = gw_FL_FR_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                         determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                         int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_F2 and ID_BL) and not(ID_FR or ID_B or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FL_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_BL_m1
                        else:
                            gw = gw_F1_F2_FL_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_B and ID_BL) and not(ID_FR or ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FL_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_B1_BL_m1
                        else:
                            gw = gw_F1_F2_FL_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                                determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                                int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_B and ID_BL and ID_BR) and not(ID_FR or ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_FL_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_B1_BL_BR_m1
                        else:
                            gw = gw_F1_F2_FL_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                                determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                                int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_B and ID_BL) and not(ID_FR or ID_B2 or ID_BR or ID_F2)):
                        add_vids_to_list(list_vids,'F1_FL_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_B1_BL_m1
                        else:
                            gw = gw_F1_FL_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                                determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                                int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_B) and not(ID_FR or ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FL_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_B1_BR_m1
                        else:
                            gw = gw_F1_F2_FL_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_B) and not(ID_FR or ID_B2 or ID_BL or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FL_B1',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_B1_m1
                        else:
                            gw = gw_F1_F2_FL_B1_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_B and ID_FR ) and not(ID_B2 or ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FL_FR_B1_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_B1_BR_m1
                        else:
                            gw = gw_F1_F2_FL_FR_B1_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_F2 and ID_BL and ID_B and ID_FR ) and not(ID_B2 or ID_BR)):
                        add_vids_to_list(list_vids,'F1_F2_FL_FR_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_B1_BL_m1
                        else:
                            gw = gw_F1_F2_FL_FR_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_BL and ID_B and ID_FR ) and not(ID_B2 or ID_BR or ID_F2)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_BL_m1
                        else:
                            gw = gw_F1_FL_FR_B1_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_BL and ID_B and ID_FR and ID_B2) and not(ID_BR or ID_F2)):
                        add_vids_to_list(list_vids,'F1_FL_FR_B1_B2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_FR_B1_BL_B2_m1
                        else:
                            gw = gw_F1_FL_FR_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                                determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                                int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True

                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_FR ) and not(ID_B2 or ID_BL or ID_B)):
                        add_vids_to_list(list_vids,'F1_F2_FL_FR_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_BR_m1
                        else:
                            gw = gw_F1_F2_FL_FR_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_B and ID_FR and ID_B2 ) and not( ID_BL)):
                        add_vids_to_list(list_vids,'F1_F2_FL_FR_B1_B2_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_B1_B2_BR_m1
                        else:
                            gw = gw_F1_F2_FL_FR_B1_B2_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                         determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                         int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_BR and ID_B and ID_FR and ID_BL ) and not( ID_B2)):
                        add_vids_to_list(list_vids,'F1_F2_FL_FR_B1_BL_BR',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        FR_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'x'].iloc[0]
                        FR_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FR]
                        if time+120 in df_obs['Time'].values:
                            FR_n_x=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'x'].iloc[0]
                            FR_n_y=df.loc[((df['ID']==ID_FR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FR_n_x,FR_n_y=FR_x,FR_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        BR_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'x'].iloc[0]
                        BR_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BR]
                        if time+120 in df_obs['Time'].values:
                            BR_n_x=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'x'].iloc[0]
                            BR_n_y=df.loc[((df['ID']==ID_BR) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BR_n_x,BR_n_y=BR_x,BR_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_FR_B1_BL_BR_m1
                        else:
                            gw = gw_F1_F2_FL_FR_B1_BL_BR_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BR_x,BR_y))),
                                                determine_action(BR_x,BR_y,BR_n_x,BR_n_y),
                                                int(gw.point_to_int((BR_n_x,BR_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FR_x,FR_y))),
                                                determine_action(FR_x,FR_y,FR_n_x,FR_n_y),
                                                int(gw.point_to_int((FR_n_x,FR_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_F2 and ID_B2 and ID_B and ID_BL ) and not( ID_BR or ID_FR)):
                        add_vids_to_list(list_vids,'F1_F2_FL_B1_B2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        F2_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'x'].iloc[0]
                        F2_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F2]
                        if time+120 in df_obs['Time'].values:
                            F2_n_x=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'x'].iloc[0]
                            F2_n_y=df.loc[((df['ID']==ID_F2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F2_n_x,F2_n_y=F2_x,F2_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_F2_FL_B1_B2_BL_m1
                        else:
                            gw = gw_F1_F2_FL_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                                determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                                int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F2_x,F2_y))),
                                         determine_action(F2_x,F2_y,F2_n_x,F2_n_y),
                                         int(gw.point_to_int((F2_n_x,F2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
                    elif ((ID_F and ID_FL and ID_B2 and ID_B and ID_BL ) and not( ID_BR or ID_FR or ID_F2 )):
                        add_vids_to_list(list_vids,'F1_FL_B1_B2_BL',v_id)
                        F1_x=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'x'].iloc[0]
                        F1_y=df.loc[((df['ID']==ID_F) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_F]
                        if time+120 in df_obs['Time'].values:
                            F1_n_x=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'x'].iloc[0]
                            F1_n_y=df.loc[((df['ID']==ID_F) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            F1_n_x,F1_n_y=F1_x,F1_y
                        FL_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'x'].iloc[0]
                        FL_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_FL]
                        if time+120 in df_obs['Time'].values:
                            FL_n_x=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'x'].iloc[0]
                            FL_n_y=df.loc[((df['ID']==ID_FL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            FL_n_x,FL_n_y=FL_x,FL_y
                        B1_x=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'x'].iloc[0]
                        B1_y=df.loc[((df['ID']==ID_B) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B]
                        if time+120 in df_obs['Time'].values:
                            B1_n_x=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'x'].iloc[0]
                            B1_n_y=df.loc[((df['ID']==ID_B) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B1_n_x,B1_n_y=B1_x,B1_y
                        B2_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'x'].iloc[0]
                        B2_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_B2]
                        if time+120 in df_obs['Time'].values:
                            B2_n_x=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'x'].iloc[0]
                            B2_n_y=df.loc[((df['ID']==ID_B2) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            B2_n_x,B2_n_y=B2_x,B2_y
                        BL_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'x'].iloc[0]
                        BL_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time)), 'y'].iloc[0]
                        df_obs=df.loc[df['ID']==ID_BL]
                        if time+120 in df_obs['Time'].values:
                            BL_n_x=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'x'].iloc[0]
                            BL_n_y=df.loc[((df['ID']==ID_BL) & (df['Time']==time+120)), 'y'].iloc[0]
                        else:
                            BL_n_x,BL_n_y=BL_x,BL_y
                        if (x<8 and y<25):
                            gw = gw_F1_FL_B1_B2_BL_m1
                        else:
                            gw = gw_F1_FL_B1_B2_BL_m2
                        obs_trajectories.append((int(gw.point_to_int((B1_x,B1_y))),
                                         determine_action(B1_x,B1_y,B1_n_x,B1_n_y),
                                         int(gw.point_to_int((B1_n_x,B1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((BL_x,BL_y))),
                                         determine_action(BL_x,BL_y,BL_n_x,BL_n_y),
                                         int(gw.point_to_int((BL_n_x,BL_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((B2_x,B2_y))),
                                                determine_action(B2_x,B2_y,B2_n_x,B2_n_y),
                                                int(gw.point_to_int((B2_n_x,B2_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((F1_x,F1_y))),
                                         determine_action(F1_x,F1_y,F1_n_x,F1_n_y),
                                         int(gw.point_to_int((F1_n_x,F1_n_y)))))
                        obs_trajectories.append((int(gw.point_to_int((FL_x,FL_y))),
                                                determine_action(FL_x,FL_y,FL_n_x,FL_n_y),
                                                int(gw.point_to_int((FL_n_x,FL_n_y)))))
                        vehicle_found = True
############################################################################################################3
                    if vehicle_found:
                        i=  df.loc[((df['ID']==v_id) & (df['Time']==time))].index[0]
                        if (time%10000==0):
                            print('Vehicle ID:',v_id)
                            write_true = True
                        load_trajectories(gw,trajectories,df,i)
                        gw._trans_prob_irl(v_id,trajectories,obs_trajectories)
                        ground_r=np.array([gw.reward(s) for s in range(gw.n_states)]) 
                        fea_mat = gw.feature_matrix_irl(trajectories,obs_trajectories)
                        feature_matrix=f_sc.fit_transform(fea_mat)
                        GAMMA=discount #1
                        if (x<8 and y<25):
                            if ((ID_F and ID_FR and ID_FL) and not (ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_FL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_FL_m1], axis=0)
                                avg_reward_F1_FR_FL_m1,first_ego_F1_FR_FL_m1,loss_F1_FR_FL_m1 = computation(first_ego_F1_FR_FL_m1 ,avg_reward_F1_FR_FL_m1,
                                                                                              avg_grnd_rwd_F1_FR_FL_m1 ,deep_max_F1_FR_FL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_FL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_FL and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_F2_FR_FL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_FL_m1], axis=0)
                                avg_reward_F1_F2_FR_FL_m1,first_ego_F1_F2_FR_FL_m1,loss_F1_F2_FR_FL_m1 = computation(first_ego_F1_F2_FR_FL_m1 ,avg_reward_F1_F2_FR_FL_m1,
                                                                                              avg_grnd_rwd_F1_F2_FR_FL_m1 ,deep_max_F1_F2_FR_FL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_FL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_FL)):
                                avg_grnd_rwd_F1_F2_FR_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_BR_m1], axis=0)
                                avg_reward_F1_F2_FR_BR_m1,first_ego_F1_F2_FR_BR_m1,loss_F1_F2_FR_BR_m1 = computation(first_ego_F1_F2_FR_BR_m1 ,avg_reward_F1_F2_FR_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_FR_BR_m1 ,deep_max_F1_F2_FR_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_B and ID_F2) and not (ID_B2 or ID_BL or ID_FL)):
                                avg_grnd_rwd_F1_F2_FR_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BR_m1], axis=0)
                                avg_reward_F1_F2_FR_B1_BR_m1,first_ego_F1_F2_FR_B1_BR_m1,loss_F1_F2_FR_B1_BR_m1 = computation(first_ego_F1_F2_FR_B1_BR_m1 ,avg_reward_F1_F2_FR_B1_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_FR_B1_BR_m1 ,deep_max_F1_F2_FR_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_B) and not (ID_B2 or ID_BL or ID_FL or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BR_m1], axis=0)
                                avg_reward_F1_FR_B1_BR_m1,first_ego_F1_FR_B1_BR_m1,loss_F1_FR_B1_BR_m1 = computation(first_ego_F1_FR_B1_BR_m1 ,avg_reward_F1_FR_B1_BR_m1,
                                                                                              avg_grnd_rwd_F1_FR_B1_BR_m1 ,deep_max_F1_FR_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL) and not (ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_F2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_F2_m1], axis=0)  
                                avg_reward_F1_FL_F2_m1,first_ego_F1_FL_F2_m1,loss_F1_FL_F2_m1 = computation(first_ego_F1_FL_F2_m1,avg_reward_F1_FL_F2_m1,
                                                                                              avg_grnd_rwd_F1_FL_F2_m1,deep_max_F1_FL_F2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FL_F2_m1',write_true)
                                write_true = False 
                            elif ((ID_F and ID_F2 and ID_FR) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_F2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_F2_m1], axis=0)  
                                avg_reward_F1_FR_F2_m1,first_ego_F1_FR_F2_m1,loss_F1_FR_F2_m1 = computation(first_ego_F1_FR_F2_m1,avg_reward_F1_FR_F2_m1,
                                                                                              avg_grnd_rwd_F1_FR_F2_m1,deep_max_F1_FR_F2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FR_F2_m1',write_true)
                                write_true = False  
                            elif ((ID_F and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                                avg_grnd_rwd_F1_F2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_m1], axis=0)  
                                avg_reward_F1_F2_m1,first_ego_F1_F2_m1,loss_F1_F2_m1 = computation(first_ego_F1_F2_m1,avg_reward_F1_F2_m1,
                                                                                              avg_grnd_rwd_F1_F2_m1,deep_max_F1_F2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_F2__m1',write_true)
                                write_true = False  
                            elif ((ID_F and ID_FR) and not (ID_F2 and ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_m1], axis=0)  
                                avg_reward_F1_FR_m1,first_ego_F1_FR_m1,loss_F1_FR_m1 = computation(first_ego_F1_FR_m1,avg_reward_F1_FR_m1,
                                                                                              avg_grnd_rwd_F1_FR_m1,deep_max_F1_FR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FR__m1',write_true)
                                write_true = False 
                            elif ((ID_FL and ID_FR) and not (ID_F2 and ID_F or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_FL_FR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_FR_m1], axis=0)  
                                avg_reward_FL_FR_m1,first_ego_FL_FR_m1,loss_FL_FR_m1 = computation(first_ego_FL_FR_m1,avg_reward_FL_FR_m1,
                                                                                              avg_grnd_rwd_FL_FR_m1,deep_max_FL_FR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'FL_FR__m1',write_true)
                                write_true = False 
                            elif ((ID_F and ID_FL) and not (ID_F2 and ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_m1], axis=0)  
                                avg_reward_F1_FL_m1,first_ego_F1_FL_m1,loss_F1_FL_m1 = computation(first_ego_F1_FL_m1,avg_reward_F1_FL_m1,
                                                                                              avg_grnd_rwd_F1_FL_m1,deep_max_F1_FL_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FL_m1',write_true)
                                write_true = False 
                            elif (ID_FL and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                                avg_grnd_rwd_FL_m1 = np.mean([ground_r, avg_grnd_rwd_FL_m1], axis=0)  
                                avg_reward_FL_m1,first_ego_FL_m1,loss_FL_m1 = computation(first_ego_FL_m1,avg_reward_FL_m1,
                                                                                              avg_grnd_rwd_FL_m1,deep_max_FL_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'FL_m1',write_true)
                                write_true = False
                            elif (ID_FR and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_FR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_m1], axis=0)  
                                avg_reward_FR_m1,first_ego_FR_m1,loss_FR_m1 = computation(first_ego_FR_m1,avg_reward_FR_m1,
                                                                                              avg_grnd_rwd_FR_m1,deep_max_FR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'FR_m1',write_true)
                                write_true = False
                            elif (ID_F and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_m1 = np.mean([ground_r, avg_grnd_rwd_F1_m1], axis=0)  
                                avg_reward_F1_m1,first_ego_F1_m1,loss_F1_m1 = computation(first_ego_F1_m1,avg_reward_F1_m1,
                                                                                              avg_grnd_rwd_F1_m1,deep_max_F1_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B2 or ID_BL)):
                                avg_grnd_rwd_F1_FL_FR_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_BR_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_BR_m1,first_ego_F1_FL_FR_B1_BR_m1,loss_F1_FL_FR_B1_BR_m1 = computation(first_ego_F1_FL_FR_B1_BR_m1,                                           avg_reward_F1_FL_FR_B1_BR_m1,avg_grnd_rwd_F1_FL_FR_B1_BR_m1,deep_max_F1_FL_FR_B1_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_BR_m1',write_true)
                                write_true = False
                            #print('Vehicle ID with FL')
                            #print('ID_FL:',ID_FL,ID_F,ID_F2, ID_B, ID_B2,ID_BL,ID_BR, ID_FR)
############################################################################################################
                            elif ((ID_B and ID_BR and ID_BL) and not (ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_BL_m1 = np.mean([ground_r, avg_grnd_rwd_B1_BR_BL_m1], axis=0)
                                avg_reward_B1_BR_BL_m1,first_ego_B1_BR_BL_m1,loss_B1_BR_BL_m1 = computation(first_ego_B1_BR_BL_m1 ,avg_reward_B1_BR_BL_m1,
                                                                                              avg_grnd_rwd_B1_BR_BL_m1 ,deep_max_B1_BR_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'B1_BR_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_BL and ID_B2) and not (ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_B2_BR_BL_m1 = np.mean([ground_r, avg_grnd_rwd_B1_B2_BR_BL_m1], axis=0)
                                avg_reward_B1_B2_BR_BL_m1,first_ego_B1_B2_BR_BL_m1,loss_B1_B2_BR_BL_m1 = computation(first_ego_B1_B2_BR_BL_m1 ,avg_reward_B1_B2_BR_BL_m1,
                                                                                              avg_grnd_rwd_B1_B2_BR_BL_m1 ,deep_max_B1_B2_BR_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'B1_B2_BR_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BL_B2_m1 = np.mean([ground_r, avg_grnd_rwd_B1_BL_B2_m1], axis=0)  
                                avg_reward_B1_BL_B2_m1,first_ego_B1_BL_B2_m1,loss_B1_BL_B2_m1 = computation(first_ego_B1_BL_B2_m1,avg_reward_B1_BL_B2_m1,
                                                                                              avg_grnd_rwd_B1_BL_B2_m1,deep_max_B1_BL_B2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BL_B2_m1',write_true)
                                write_true = False 
                            elif ((ID_B and ID_B2 and ID_BR) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_B2_m1 = np.mean([ground_r, avg_grnd_rwd_B1_BR_B2_m1], axis=0)  
                                avg_reward_B1_BR_B2_m1,first_ego_B1_BR_B2_m1,loss_B1_BR_B2_m1 = computation(first_ego_B1_BR_B2_m1,avg_reward_B1_BR_B2_m1,
                                                                                              avg_grnd_rwd_B1_BR_B2_m1,deep_max_B1_BR_B2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BR_B2_m1',write_true)
                                write_true = False  
                            elif ((ID_B and ID_B2) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_B1_B2_m1], axis=0)  
                                avg_reward_B1_B2_m1,first_ego_B1_B2_m1,loss_B1_B2_m1 = computation(first_ego_B1_B2_m1,avg_reward_B1_B2_m1,
                                                                                              avg_grnd_rwd_B1_B2_m1,deep_max_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_B2__m1',write_true)
                                write_true = False  
                            elif ((ID_B and ID_BR) and not (ID_B2 and ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_B1_BR_m1], axis=0)  
                                avg_reward_B1_BR_m1,first_ego_B1_BR_m1,loss_B1_BR_m1 = computation(first_ego_B1_BR_m1,avg_reward_B1_BR_m1,
                                                                                              avg_grnd_rwd_B1_BR_m1,deep_max_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BR__m1',write_true)
                                write_true = False 
                            elif ((ID_BL and ID_BR) and not (ID_B2 and ID_B or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_BL_BR_m1], axis=0)  
                                avg_reward_BL_BR_m1,first_ego_BL_BR_m1,loss_BL_BR_m1 = computation(first_ego_BL_BR_m1,avg_reward_BL_BR_m1,
                                                                                              avg_grnd_rwd_BL_BR_m1,deep_max_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'BL_BR__m1',write_true)
                                write_true = False 
                            elif ((ID_B and ID_BL) and not (ID_B2 and ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_B1_BL_m1], axis=0)  
                                avg_reward_B1_BL_m1,first_ego_B1_BL_m1,loss_B1_BL_m1 = computation(first_ego_B1_BL_m1,avg_reward_B1_BL_m1,
                                                                                              avg_grnd_rwd_B1_BL_m1,deep_max_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BL_m1',write_true)
                                write_true = False 
                            elif (ID_BL and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_BL_m1 = np.mean([ground_r, avg_grnd_rwd_BL_m1], axis=0)  
                                avg_reward_BL_m1,first_ego_BL_m1,loss_BL_m1 = computation(first_ego_BL_m1,avg_reward_BL_m1,
                                                                                              avg_grnd_rwd_BL_m1,deep_max_BL_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'BL_m1',write_true)
                                write_true = False
                            elif (ID_BR and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_BR_m1 = np.mean([ground_r, avg_grnd_rwd_BR_m1], axis=0)  
                                avg_reward_BR_m1,first_ego_BR_m1,loss_BR_m1 = computation(first_ego_BR_m1,avg_reward_BR_m1,
                                                                                              avg_grnd_rwd_BR_m1,deep_max_BR_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'BR_m1',write_true)
                                write_true = False
                            elif (ID_B and not (ID_BR or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_B1_m1 = np.mean([ground_r, avg_grnd_rwd_B1_m1], axis=0)  
                                avg_reward_B1_m1,first_ego_B1_m1,loss_B1_m1 = computation(first_ego_B1_m1,avg_reward_B1_m1,
                                                                                              avg_grnd_rwd_B1_m1,deep_max_B1_m1,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_m1',write_true)
                                write_true = False
#######################################################################################################################3
                            elif ((ID_B and ID_F) and not (ID_BR or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_m1], axis=0)
                                avg_reward_F1_B1_m1,first_ego_F1_B1_m1,loss_F1_B1_m1 = computation(first_ego_F1_B1_m1,avg_reward_F1_B1_m1,
                                                                                              avg_grnd_rwd_F1_B1_m1,deep_max_F1_B1_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_F) and not (ID_B or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_BR_m1], axis=0)
                                avg_reward_F1_BR_m1,first_ego_F1_BR_m1,loss_F1_BR_m1 = computation(first_ego_F1_BR_m1,avg_reward_F1_BR_m1,
                                                                                              avg_grnd_rwd_F1_BR_m1,deep_max_F1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BL and ID_F) and not (ID_B or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_BL_m1], axis=0)
                                avg_reward_F1_BL_m1,first_ego_F1_BL_m1,loss_F1_BL_m1 = computation(first_ego_F1_BL_m1,avg_reward_F1_BL_m1,
                                                                                              avg_grnd_rwd_F1_BL_m1,deep_max_F1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_B2) and not (ID_BR or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_m1], axis=0)
                                avg_reward_F1_B1_B2_m1,first_ego_F1_B1_B2_m1,loss_F1_B1_B2_m1 = computation(first_ego_F1_B1_B2_m1,avg_reward_F1_B1_B2_m1,
                                                                                              avg_grnd_rwd_F1_B1_B2_m1,deep_max_F1_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BR) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BR_m1], axis=0)
                                avg_reward_F1_B1_BR_m1,first_ego_F1_B1_BR_m1,loss_F1_B1_BR_m1 = computation(first_ego_F1_B1_BR_m1,avg_reward_F1_B1_BR_m1,
                                                                                              avg_grnd_rwd_F1_B1_BR_m1,deep_max_F1_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BL_m1], axis=0)
                                avg_reward_F1_B1_BL_m1,first_ego_F1_B1_BL_m1,loss_F1_B1_BL_m1 = computation(first_ego_F1_B1_BL_m1,avg_reward_F1_B1_BL_m1,
                                                                                              avg_grnd_rwd_F1_B1_BL_m1,deep_max_F1_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BL_m1',write_true)
                                write_true = False

                            elif ((ID_BR and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_B)):
                                avg_grnd_rwd_F1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_BL_BR_m1], axis=0)
                                avg_reward_F1_BL_BR_m1,first_ego_F1_BL_BR_m1,loss_F1_BL_BR_m1 = computation(first_ego_F1_BL_BR_m1,avg_reward_F1_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_BL_BR_m1,deep_max_F1_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_B1_BL_BR_m1,first_ego_F1_B1_BL_BR_m1,loss_F1_B1_BL_BR_m1 = computation(first_ego_F1_B1_BL_BR_m1,avg_reward_F1_B1_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_B1_BL_BR_m1,deep_max_F1_B1_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_F and ID_BL) and not (ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_F1_B1_B2_BL_BR_m1,first_ego_F1_B1_B2_BL_BR_m1,loss_F1_B1_B2_BL_BR_m1 = computation(first_ego_F1_B1_B2_BL_BR_m1,avg_reward_F1_B1_B2_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_B1_B2_BL_BR_m1,deep_max_F1_B1_B2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_B2_BL_BR_m1',write_true)
                                write_true = False
#################################################################################################### FL_combination
                            elif ((ID_B and ID_FL) and not (ID_BR or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_m1], axis=0)
                                avg_reward_FL_B1_m1,first_ego_FL_B1_m1,loss_FL_B1_m1 = computation(first_ego_FL_B1_m1,avg_reward_FL_B1_m1,
                                                                                              avg_grnd_rwd_FL_B1_m1,deep_max_FL_B1_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_FL) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_BR_m1], axis=0)
                                avg_reward_FL_BR_m1,first_ego_FL_BR_m1,loss_FL_BR_m1 = computation(first_ego_FL_BR_m1,avg_reward_FL_BR_m1,
                                                                                              avg_grnd_rwd_FL_BR_m1,deep_max_FL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BL and ID_FL) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FL_BL_m1], axis=0)
                                avg_reward_FL_BL_m1,first_ego_FL_BL_m1,loss_FL_BL_m1 = computation(first_ego_FL_BL_m1,avg_reward_FL_BL_m1,
                                                                                              avg_grnd_rwd_FL_BL_m1,deep_max_FL_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_B2) and not (ID_BR or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_m1], axis=0)
                                avg_reward_FL_B1_B2_m1,first_ego_FL_B1_B2_m1,loss_FL_B1_B2_m1 = computation(first_ego_FL_B1_B2_m1,avg_reward_FL_B1_B2_m1,
                                                                                              avg_grnd_rwd_FL_B1_B2_m1,deep_max_FL_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_BR) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BR_m1], axis=0)
                                avg_reward_FL_B1_BR_m1,first_ego_FL_B1_BR_m1,loss_FL_B1_BR_m1 = computation(first_ego_FL_B1_BR_m1,avg_reward_FL_B1_BR_m1,
                                                                                              avg_grnd_rwd_FL_B1_BR_m1,deep_max_FL_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BL_m1], axis=0)
                                avg_reward_FL_B1_BL_m1,first_ego_FL_B1_BL_m1,loss_FL_B1_BL_m1 = computation(first_ego_FL_B1_BL_m1,avg_reward_FL_B1_BL_m1,
                                                                                              avg_grnd_rwd_FL_B1_BL_m1,deep_max_FL_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BL_m1',write_true)
                                write_true = False

                            elif ((ID_BR and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_B)):
                                avg_grnd_rwd_FL_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_BL_BR_m1], axis=0)
                                avg_reward_FL_BL_BR_m1,first_ego_FL_BL_BR_m1,loss_FL_BL_BR_m1 = computation(first_ego_FL_BL_BR_m1,avg_reward_FL_BL_BR_m1,
                                                                                              avg_grnd_rwd_FL_BL_BR_m1,deep_max_FL_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR)):
                                avg_grnd_rwd_FL_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BL_BR_m1], axis=0)
                                avg_reward_FL_B1_BL_BR_m1,first_ego_FL_B1_BL_BR_m1,loss_FL_B1_BL_BR_m1 = computation(first_ego_FL_B1_BL_BR_m1,avg_reward_FL_B1_BL_BR_m1,
                                                                                              avg_grnd_rwd_FL_B1_BL_BR_m1,deep_max_FL_B1_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_FL and ID_BL) and not (ID_F2 or ID_F or ID_FR)):
                                avg_grnd_rwd_FL_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_FL_B1_B2_BL_BR_m1,first_ego_FL_B1_B2_BL_BR_m1,loss_FL_B1_B2_BL_BR_m1 = computation(first_ego_FL_B1_B2_BL_BR_m1,avg_reward_FL_B1_B2_BL_BR_m1,
                                                                                              avg_grnd_rwd_FL_B1_B2_BL_BR_m1,deep_max_FL_B1_B2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B2 and ID_B and ID_FL and ID_BL) and not (ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_BL_m1], axis=0)
                                avg_reward_FL_B1_B2_BL_m1,first_ego_FL_B1_B2_BL_m1,loss_FL_B1_B2_BL_m1 = computation(first_ego_FL_B1_B2_BL_m1,avg_reward_FL_B1_B2_BL_m1,
                                                                                              avg_grnd_rwd_FL_B1_B2_BL_m1,deep_max_FL_B1_B2_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_BL_m1',write_true)
                                write_true = False
################################################################################################## FR combination
                            elif ((ID_B and ID_FR) and not (ID_BR or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_m1], axis=0)
                                avg_reward_FR_B1_m1,first_ego_FR_B1_m1,loss_FR_B1_m1 = computation(first_ego_FR_B1_m1,avg_reward_FR_B1_m1,
                                                                                              avg_grnd_rwd_FR_B1_m1,deep_max_FR_B1_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_FR) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_BR_m1], axis=0)
                                avg_reward_FR_BR_m1,first_ego_FR_BR_m1,loss_FR_BR_m1 = computation(first_ego_FR_BR_m1,avg_reward_FR_BR_m1,
                                                                                              avg_grnd_rwd_FR_BR_m1,deep_max_FR_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BL and ID_FR) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BR)):
                                avg_grnd_rwd_FR_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FR_BL_m1], axis=0)
                                avg_reward_FR_BL_m1,first_ego_FR_BL_m1,loss_FR_BL_m1 = computation(first_ego_FR_BL_m1,avg_reward_FR_BL_m1,
                                                                                              avg_grnd_rwd_FR_BL_m1,deep_max_FR_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_B2) and not (ID_BR or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_m1], axis=0)
                                avg_reward_FR_B1_B2_m1,first_ego_FR_B1_B2_m1,loss_FR_B1_B2_m1 = computation(first_ego_FR_B1_B2_m1,avg_reward_FR_B1_B2_m1,
                                                                                              avg_grnd_rwd_FR_B1_B2_m1,deep_max_FR_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_BR) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BR_m1], axis=0)
                                avg_reward_FR_B1_BR_m1,first_ego_FR_B1_BR_m1,loss_FR_B1_BR_m1 = computation(first_ego_FR_B1_BR_m1,avg_reward_FR_B1_BR_m1,
                                                                                              avg_grnd_rwd_FR_B1_BR_m1,deep_max_FR_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_BR)):
                                avg_grnd_rwd_FR_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BL_m1], axis=0)
                                avg_reward_FR_B1_BL_m1,first_ego_FR_B1_BL_m1,loss_FR_B1_BL_m1 = computation(first_ego_FR_B1_BL_m1,avg_reward_FR_B1_BL_m1,
                                                                                              avg_grnd_rwd_FR_B1_BL_m1,deep_max_FR_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BL_m1',write_true)
                                write_true = False

                            elif ((ID_BR and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_B)):
                                avg_grnd_rwd_FR_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_BL_BR_m1], axis=0)
                                avg_reward_FR_BL_BR_m1,first_ego_FR_BL_BR_m1,loss_FR_BL_BR_m1 = computation(first_ego_FR_BL_BR_m1,avg_reward_FR_BL_BR_m1,
                                                                                              avg_grnd_rwd_FR_BL_BR_m1,deep_max_FR_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL)):
                                avg_grnd_rwd_FR_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BL_BR_m1], axis=0)
                                avg_reward_FR_B1_BL_BR_m1,first_ego_FR_B1_BL_BR_m1,loss_FR_B1_BL_BR_m1 = computation(first_ego_FR_B1_BL_BR_m1,avg_reward_FR_B1_BL_BR_m1,
                                                                                              avg_grnd_rwd_FR_B1_BL_BR_m1,deep_max_FR_B1_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_FR and ID_BL) and not (ID_F2 or ID_F or ID_FL)):
                                avg_grnd_rwd_FR_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_FR_B1_B2_BL_BR_m1,first_ego_FR_B1_B2_BL_BR_m1,loss_FR_B1_B2_BL_BR_m1 = computation(first_ego_FR_B1_B2_BL_BR_m1,avg_reward_FR_B1_B2_BL_BR_m1,
                                                                                              avg_grnd_rwd_FR_B1_B2_BL_BR_m1,deep_max_FR_B1_B2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_BL_BR_m1',write_true)
                                write_true = False
################################################################################################### F1_F2_combination
                            elif ((ID_B and ID_F and ID_F2) and not (ID_BR or ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_m1], axis=0)
                                avg_reward_F1_F2_B1_m1,first_ego_F1_F2_B1_m1,loss_F1_F2_B1_m1 = computation(first_ego_F1_F2_B1_m1,avg_reward_F1_F2_B1_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_m1,deep_max_F1_F2_B1_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_F and ID_F2) and not (ID_B or ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BR_m1], axis=0)
                                avg_reward_F1_F2_BR_m1,first_ego_F1_F2_BR_m1,loss_F1_F2_BR_m1 = computation(first_ego_F1_F2_BR_m1,avg_reward_F1_F2_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_BR_m1,deep_max_F1_F2_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BL and ID_F and ID_F2) and not (ID_B or ID_B2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BL_m1], axis=0)
                                avg_reward_F1_F2_BL_m1,first_ego_F1_F2_BL_m1,loss_F1_F2_BL_m1 = computation(first_ego_F1_F2_BL_m1,avg_reward_F1_F2_BL_m1,
                                                                                              avg_grnd_rwd_F1_F2_BL_m1,deep_max_F1_F2_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_B2 and ID_F2) and not (ID_BR or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_m1], axis=0)
                                avg_reward_F1_F2_B1_B2_m1,first_ego_F1_F2_B1_B2_m1,loss_F1_F2_B1_B2_m1 = computation(first_ego_F1_F2_B1_B2_m1,avg_reward_F1_F2_B1_B2_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_m1,deep_max_F1_F2_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BR and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BR_m1], axis=0)
                                avg_reward_F1_F2_B1_BR_m1,first_ego_F1_F2_B1_BR_m1,loss_F1_F2_B1_BR_m1 = computation(first_ego_F1_F2_B1_BR_m1,avg_reward_F1_F2_B1_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_BR_m1,deep_max_F1_F2_B1_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BL_m1], axis=0)
                                avg_reward_F1_F2_B1_BL_m1,first_ego_F1_F2_B1_BL_m1,loss_F1_F2_B1_BL_m1 = computation(first_ego_F1_F2_B1_BL_m1,avg_reward_F1_F2_B1_BL_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_BL_m1,deep_max_F1_F2_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BL_m1',write_true)
                                write_true = False

                            elif ((ID_BR and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_B)):
                                avg_grnd_rwd_F1_F2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_BL_BR_m1,first_ego_F1_F2_BL_BR_m1,loss_F1_F2_BL_BR_m1 = computation(first_ego_F1_F2_BL_BR_m1,avg_reward_F1_F2_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_BL_BR_m1,deep_max_F1_F2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_F2_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_B1_BL_BR_m1,first_ego_F1_F2_B1_BL_BR_m1,loss_F1_F2_B1_BL_BR_m1 = computation(first_ego_F1_F2_B1_BL_BR_m1,avg_reward_F1_F2_B1_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_BL_BR_m1,deep_max_F1_F2_B1_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_F and ID_BL and ID_F2) and not (ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_B1_B2_BL_BR_m1,first_ego_F1_F2_B1_B2_BL_BR_m1,loss_F1_F2_B1_B2_BL_BR_m1 = computation(first_ego_F1_F2_B1_B2_BL_BR_m1,avg_reward_F1_F2_B1_B2_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m1,deep_max_F1_F2_B1_B2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B2 and ID_B and ID_F and ID_BL and ID_F2) and not (ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_BL_m1], axis=0)
                                avg_reward_F1_F2_B1_B2_BL_m1,first_ego_F1_F2_B1_B2_BL_m1,loss_F1_F2_B1_B2_BL_m1 = computation(first_ego_F1_F2_B1_B2_BL_m1,avg_reward_F1_F2_B1_B2_BL_m1,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_BL_m1,deep_max_F1_F2_B1_B2_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_BL_m1',write_true)
                                write_true = False
#####################################################################################################################
                            elif (ID_B and ID_BR and ID_B2 and ID_F and ID_F2 and ID_FL and ID_FR and ID_BL):
                                avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m1], axis=0)
                                avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m1,first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m1,loss_F1_F2_FR_FL_B1_B2_BR_BL_m1 = computation(first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m1,
                                                                                                                                                         avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m1,
                                                                                              avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m1,deep_max_F1_F2_FR_FL_B1_B2_BR_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_FL_B1_B2_BR_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR and ID_BL) and not (ID_F2)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BL_BR_m1,first_ego_F1_FL_FR_B1_B2_BL_BR_m1,loss_F1_FL_FR_B1_B2_BL_BR_m1 = computation(first_ego_F1_FL_FR_B1_B2_BL_BR_m1,avg_reward_F1_FL_FR_B1_B2_BL_BR_m1,
                                                                                              avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m1,deep_max_F1_FL_FR_B1_B2_BL_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FL_FR_B1_B2_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_BL)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BR_m1,first_ego_F1_FL_FR_B1_B2_BR_m1,loss_F1_FL_FR_B1_B2_BR_m1 = computation(first_ego_F1_FL_FR_B1_B2_BR_m1,avg_reward_F1_FL_FR_B1_B2_BR_m1,
                                                                                              avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m1,deep_max_F1_FL_FR_B1_B2_BR_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FL_FR_B1_B2_BR_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_m1,first_ego_F1_FL_FR_B1_B2_m1,loss_F1_FL_FR_B1_B2_m1 = computation(first_ego_F1_FL_FR_B1_B2_m1,avg_reward_F1_FL_FR_B1_B2_m1,
                                                                                              avg_grnd_rwd_F1_FL_FR_B1_B2_m1,deep_max_F1_FL_FR_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FL_FR_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_F and ID_FR) and not (ID_F2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_FR_B1_B2_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_B2_m1], axis=0)
                                avg_reward_F1_FR_B1_B2_m1,first_ego_F1_FR_B1_B2_m1,loss_F1_FR_B1_B2_m1 = computation(first_ego_F1_FR_B1_B2_m1,avg_reward_F1_FR_B1_B2_m1,
                                                                                              avg_grnd_rwd_F1_FR_B1_B2_m1,deep_max_F1_FR_B1_B2_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_B2_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BL and ID_F and ID_FR) and not (ID_F2 or ID_B2 or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_FR_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BL_m1], axis=0)
                                avg_reward_F1_FR_B1_BL_m1,first_ego_F1_FR_B1_BL_m1,loss_F1_FR_B1_BL_m1 = computation(first_ego_F1_FR_B1_BL_m1,avg_reward_F1_FR_B1_BL_m1,
                                                                                              avg_grnd_rwd_F1_FR_B1_BL_m1,deep_max_F1_FR_B1_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_B and ID_BL and  ID_B2 and ID_FR) and not (ID_F2 or ID_BL or ID_BR or ID_FL or ID_F)):
                                avg_grnd_rwd_FR_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_BL_m1], axis=0)
                                avg_reward_FR_B1_B2_BL_m1,first_ego_FR_B1_B2_BL_m1,loss_FR_B1_B2_BL_m1 = computation(first_ego_FR_B1_B2_BL_m1,avg_reward_FR_B1_B2_BL_m1,
                                                                                              avg_grnd_rwd_FR_B1_B2_BL_m1,deep_max_FR_B1_B2_BL_m1,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BR and ID_FR) and not (ID_B2 or ID_BL or ID_B)):
                                avg_grnd_rwd_F1_F2_FL_FR_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_BR_m1,first_ego_F1_F2_FL_FR_BR_m1,loss_F1_F2_FL_FR_BR_m1 = computation(first_ego_F1_F2_FL_FR_BR_m1,avg_reward_F1_F2_FL_FR_BR_m1,avg_grnd_rwd_F1_F2_FL_FR_BR_m1,deep_max_F1_F2_FL_FR_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_BR_m1',write_true)
                                write_true = False
######################################################################################################################
                            elif ((ID_F and ID_F2 and ID_FL and ID_BR) and not (ID_B or ID_B2 or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_BR_m1,first_ego_F1_F2_FL_BR_m1,loss_F1_F2_FL_BR_m1 = computation(first_ego_F1_F2_FL_BR_m1,avg_reward_F1_F2_FL_BR_m1,avg_grnd_rwd_F1_F2_FL_BR_m1,deep_max_F1_F2_FL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BL) and not (ID_B or ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BL_m1], axis=0)
                                avg_reward_F1_F2_FL_BL_m1,first_ego_F1_F2_FL_BL_m1,loss_F1_F2_FL_BL_m1 = computation(first_ego_F1_F2_FL_BL_m1,avg_reward_F1_F2_FL_BL_m1,avg_grnd_rwd_F1_F2_FL_BL_m1,deep_max_F1_F2_FL_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BL_m1',write_true)
                                write_true = False

                            elif ((ID_F and ID_F2 and ID_FL and ID_BL and ID_BR) and not (ID_B or ID_B2 or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_BL_BR_m1,first_ego_F1_F2_FL_BL_BR_m1,loss_F1_F2_FL_BL_BR_m1 = computation(first_ego_F1_F2_FL_BL_BR_m1,avg_reward_F1_F2_FL_BL_BR_m1,avg_grnd_rwd_F1_F2_FL_BL_BR_m1,deep_max_F1_F2_FL_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BL and ID_BR and ID_FR) and not (ID_B or ID_B2)):
                                avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_BL_BR_m1,first_ego_F1_F2_FL_FR_BL_BR_m1,loss_F1_F2_FL_FR_BL_BR_m1 = computation(first_ego_F1_F2_FL_FR_BL_BR_m1,avg_reward_F1_F2_FL_FR_BL_BR_m1,avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m1,deep_max_F1_F2_FL_FR_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_FL and ID_BL and ID_FR and ID_B and ID_B2 ) and not (ID_F or ID_F2 or ID_BR)):
                                avg_grnd_rwd_FL_FR_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_FL_FR_B1_B2_BL_m1], axis=0)
                                avg_reward_FL_FR_B1_B2_BL_m1,first_ego_FL_FR_B1_B2_BL_m1,loss_FL_FR_B1_B2_BL_m1 = computation(first_ego_FL_FR_B1_B2_BL_m1,avg_reward_FL_FR_B1_B2_BL_m1,avg_grnd_rwd_FL_FR_B1_B2_BL_m1,deep_max_FL_FR_B1_B2_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'FL_FR_B1_B2_BL_m1',write_true)
                                write_true = False
					
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL) and not (ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BL_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_BL_m1,first_ego_F1_F2_FL_B1_BL_m1,loss_F1_F2_FL_B1_BL_m1 = computation(first_ego_F1_F2_FL_B1_BL_m1,avg_reward_F1_F2_FL_B1_BL_m1,avg_grnd_rwd_F1_F2_FL_B1_BL_m1,deep_max_F1_F2_FL_B1_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL and ID_BR) and not (ID_B2 or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_BL_BR_m1,first_ego_F1_F2_FL_B1_BL_BR_m1,loss_F1_F2_FL_B1_BL_BR_m1 = computation(first_ego_F1_F2_FL_B1_BL_BR_m1,avg_reward_F1_F2_FL_B1_BL_BR_m1,avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m1,deep_max_F1_F2_FL_B1_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_BL) and not (ID_F2 or ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_FL_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_B1_BL_m1], axis=0)
                                avg_reward_F1_FL_B1_BL_m1,first_ego_F1_FL_B1_BL_m1,loss_F1_FL_B1_BL_m1 = computation(first_ego_F1_FL_B1_BL_m1,avg_reward_F1_FL_B1_BL_m1,avg_grnd_rwd_F1_FL_B1_BL_m1,deep_max_F1_FL_B1_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_B1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_B2 and ID_B and ID_BL) and not (ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_BL_m1], axis=0)
                                avg_reward_F1_B1_B2_BL_m1,first_ego_F1_B1_B2_BL_m1,loss_F1_B1_B2_BL_m1 = computation(first_ego_F1_B1_B2_BL_m1,avg_reward_F1_B1_B2_BL_m1,avg_grnd_rwd_F1_B1_B2_BL_m1,deep_max_F1_B1_B2_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_B1_B2_BL_m1',write_true)
                                write_true = False

                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR) and not (ID_B2 or ID_BL or ID_FR )):
                                avg_grnd_rwd_F1_F2_FL_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_BR_m1,first_ego_F1_F2_FL_B1_BR_m1,loss_F1_F2_FL_B1_BR_m1 = computation(first_ego_F1_F2_FL_B1_BR_m1,avg_reward_F1_F2_FL_B1_BR_m1,avg_grnd_rwd_F1_F2_FL_B1_BR_m1,deep_max_F1_F2_FL_B1_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B) and not (ID_B2 or ID_BL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_B1_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_m1,first_ego_F1_F2_FL_B1_m1,loss_F1_F2_FL_B1_m1 = computation(first_ego_F1_F2_FL_B1_m1,avg_reward_F1_F2_FL_B1_m1,avg_grnd_rwd_F1_F2_FL_B1_m1,deep_max_F1_F2_FL_B1_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR) and not (ID_B2 or ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BR_m1,first_ego_F1_F2_FL_FR_B1_BR_m1,loss_F1_F2_FL_FR_B1_BR_m1 = computation(first_ego_F1_F2_FL_FR_B1_BR_m1,avg_reward_F1_F2_FL_FR_B1_BR_m1,avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m1,deep_max_F1_F2_FL_FR_B1_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL and ID_FR) and not (ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BL_m1,first_ego_F1_F2_FL_FR_B1_BL_m1,loss_F1_F2_FL_FR_B1_BL_m1 = computation(first_ego_F1_F2_FL_FR_B1_BL_m1,avg_reward_F1_F2_FL_FR_B1_BL_m1,avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m1,deep_max_F1_F2_FL_FR_B1_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BL_m1',write_true)
                                write_true = False
####
                            elif ((ID_F and ID_F2 and ID_FR and ID_B and ID_BL and ID_BR) and not (ID_FL or ID_B2)):
                                avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_FR_B1_BL_BR_m1,first_ego_F1_F2_FR_B1_BL_BR_m1,loss_F1_F2_FR_B1_BL_BR_m1 = computation(first_ego_F1_F2_FR_B1_BL_BR_m1,avg_reward_F1_F2_FR_B1_BL_BR_m1,avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m1,deep_max_F1_F2_FR_B1_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FR_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FR and ID_B and ID_BL) and not (ID_FL or ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_F2_FR_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BL_m1], axis=0)
                                avg_reward_F1_F2_FR_B1_BL_m1,first_ego_F1_F2_FR_B1_BL_m1,loss_F1_F2_FR_B1_BL_m1 = computation(first_ego_F1_F2_FR_B1_BL_m1,avg_reward_F1_F2_FR_B1_BL_m1,avg_grnd_rwd_F1_F2_FR_B1_BL_m1,deep_max_F1_F2_FR_B1_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FR_B1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR) and not (ID_FL or ID_B2 or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_FR_B1_BL_BR_m1,first_ego_F1_FR_B1_BL_BR_m1,loss_F1_FR_B1_BL_BR_m1 = computation(first_ego_F1_FR_B1_BL_BR_m1,avg_reward_F1_FR_B1_BL_BR_m1,avg_grnd_rwd_F1_FR_B1_BL_BR_m1,deep_max_F1_FR_B1_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FR_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not (ID_FL or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_F1_FR_B1_B2_BL_BR_m1,first_ego_F1_FR_B1_B2_BL_BR_m1,loss_F1_FR_B1_B2_BL_BR_m1 = computation(first_ego_F1_FR_B1_B2_BL_BR_m1,avg_reward_F1_FR_B1_B2_BL_BR_m1,avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m1,deep_max_F1_FR_B1_B2_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FR_B1_B2_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_FL and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not (ID_F or ID_F2)):
                                avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m1], axis=0)
                                avg_reward_FL_FR_B1_B2_BL_BR_m1,first_ego_FL_FR_B1_B2_BL_BR_m1,loss_FL_FR_B1_B2_BL_BR_m1 = computation(first_ego_FL_FR_B1_B2_BL_BR_m1,avg_reward_FL_FR_B1_B2_BL_BR_m1,avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m1,deep_max_FL_FR_B1_B2_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'FL_FR_B1_B2_BL_BR_m1',write_true)
                                write_true = False

                            elif ((ID_F and ID_FL and ID_B and ID_BL and ID_FR) and not (ID_F2 or ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_BL_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_BL_m1,first_ego_F1_FL_FR_B1_BL_m1,loss_F1_FL_FR_B1_BL_m1 = computation(first_ego_F1_FL_FR_B1_BL_m1,avg_reward_F1_FL_FR_B1_BL_m1,avg_grnd_rwd_F1_FL_FR_B1_BL_m1,deep_max_F1_FL_FR_B1_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_BL and ID_FR and ID_B2) and not (ID_F2 or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m1], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BL_m1,first_ego_F1_FL_FR_B1_B2_BL_m1,loss_F1_FL_FR_B1_B2_BL_m1 = computation(first_ego_F1_FL_FR_B1_B2_BL_m1,avg_reward_F1_FL_FR_B1_B2_BL_m1,avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m1,deep_max_F1_FL_FR_B1_B2_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_B2_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR and ID_B2) and not (ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_B2_BR_m1,first_ego_F1_F2_FL_FR_B1_B2_BR_m1,loss_F1_F2_FL_FR_B1_B2_BR_m1 = computation(first_ego_F1_F2_FL_FR_B1_B2_BR_m1,avg_reward_F1_F2_FL_FR_B1_B2_BR_m1,avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m1,deep_max_F1_F2_FL_FR_B1_B2_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_B2_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR and ID_BL) and not (ID_B2)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m1], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BL_BR_m1,first_ego_F1_F2_FL_FR_B1_BL_BR_m1,loss_F1_F2_FL_FR_B1_BL_BR_m1 = computation(first_ego_F1_F2_FL_FR_B1_BL_BR_m1,avg_reward_F1_F2_FL_FR_B1_BL_BR_m1,avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m1,deep_max_F1_F2_FL_FR_B1_BL_BR_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BL_BR_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_B2_BL_m1,first_ego_F1_F2_FL_B1_B2_BL_m1,loss_F1_F2_FL_B1_B2_BL_m1 = computation(first_ego_F1_F2_FL_B1_B2_BL_m1,avg_reward_F1_F2_FL_B1_B2_BL_m1,avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m1,deep_max_F1_F2_FL_B1_B2_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_B2_BL_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_FR or ID_F2)):
                                avg_grnd_rwd_F1_FL_B1_B2_BL_m1 = np.mean([ground_r, avg_grnd_rwd_F1_FL_B1_B2_BL_m1], axis=0)
                                avg_reward_F1_FL_B1_B2_BL_m1,first_ego_F1_FL_B1_B2_BL_m1,loss_F1_FL_B1_B2_BL_m1 = computation(first_ego_F1_FL_B1_B2_BL_m1,avg_reward_F1_FL_B1_B2_BL_m1,avg_grnd_rwd_F1_FL_B1_B2_BL_m1,deep_max_F1_FL_B1_B2_BL_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_B1_B2_BL_m1',write_true)
                                write_true = False
######################################################################################################################
                            else:
                                avg_grnd_rwd_m1= np.mean([ground_r, avg_grnd_rwd_m1], axis=0)  
                                avg_reward_m1,first_ego_m1,loss_m1 = computation(first_ego_m1,avg_reward_m1,
                                                                                     avg_grnd_rwd_m1,deep_max_m1,
                                                                                     feature_matrix,gw, 
                                                                                     GAMMA, trajectories,obs_trajectories, 
                                                                                     learning_rate, epochs,'m1',write_true)
                                write_true = False
                            #print('m1:')
                        else:
                            if ((ID_F and ID_FR and ID_FL) and not (ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_FL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_FL_m2], axis=0)
                                avg_reward_F1_FR_FL_m2,first_ego_F1_FR_FL_m2,loss_F1_FR_FL_m2 = computation(first_ego_F1_FR_FL_m2,avg_reward_F1_FR_FL_m2,
                                                                                              avg_grnd_rwd_F1_FR_FL_m2,deep_max_F1_FR_FL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_FL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_FL and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_F2_FR_FL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_FL_m2], axis=0)
                                avg_reward_F1_F2_FR_FL_m2,first_ego_F1_F2_FR_FL_m2,loss_F1_F2_FR_FL_m2 = computation(first_ego_F1_F2_FR_FL_m2,avg_reward_F1_F2_FR_FL_m2,
                                                                                              avg_grnd_rwd_F1_F2_FR_FL_m2,deep_max_F1_F2_FR_FL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_FL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL) and not (ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_F2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_F2_m2], axis=0)  
                                avg_reward_F1_FL_F2_m2,first_ego_F1_FL_F2_m2,loss_F1_FL_F2_m2 = computation(first_ego_F1_FL_F2_m2,avg_reward_F1_FL_F2_m2,
                                                                                              avg_grnd_rwd_F1_FL_F2_m2,deep_max_F1_FL_F2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FL_F2_m2',write_true)
                                write_true = False 
                            elif ((ID_F and ID_F2 and ID_FR) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_F2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_F2_m2], axis=0)  
                                avg_reward_F1_FR_F2_m2,first_ego_F1_FR_F2_m2,loss_F1_FR_F2_m2 = computation(first_ego_F1_FR_F2_m2,avg_reward_F1_FR_F2_m2,
                                                                                              avg_grnd_rwd_F1_FR_F2_m2,deep_max_F1_FR_F2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FR_F2_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2) and not (ID_FL or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                                avg_grnd_rwd_F1_F2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_m2], axis=0)  
                                avg_reward_F1_F2_m2,first_ego_F1_F2_m2,loss_F1_F2_m2 = computation(first_ego_F1_F2_m2,avg_reward_F1_F2_m2,
                                                                                              avg_grnd_rwd_F1_F2_m2,deep_max_F1_F2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_F2_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR) and not (ID_F2 and ID_FL or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_m2], axis=0)  
                                avg_reward_F1_FR_m2,first_ego_F1_FR_m2,loss_F1_FR_m2 = computation(first_ego_F1_FR_m2,avg_reward_F1_FR_m2,
                                                                                              avg_grnd_rwd_F1_FR_m2,deep_max_F1_FR_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FR__m2',write_true)
                                write_true = False
                            elif ((ID_FL and ID_FR) and not (ID_F2 and ID_F or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_FL_FR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_FR_m2], axis=0)  
                                avg_reward_FL_FR_m2,first_ego_FL_FR_m2,loss_FL_FR_m2 = computation(first_ego_FL_FR_m2,avg_reward_FL_FR_m2,
                                                                                              avg_grnd_rwd_FL_FR_m2,deep_max_FL_FR_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'FL_FR__m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL) and not (ID_F2 and ID_FR or ID_B or ID_B2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_m2], axis=0)  
                                avg_reward_F1_FL_m2,first_ego_F1_FL_m2,loss_F1_FL_m2 = computation(first_ego_F1_FL_m2,avg_reward_F1_FL_m2,
                                                                                              avg_grnd_rwd_F1_FL_m2,deep_max_F1_FL_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'F1_FL_m2',write_true)
                                write_true = False
                            elif (ID_FL and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FR)):
                                avg_grnd_rwd_FL_m2 = np.mean([ground_r, avg_grnd_rwd_FL_m2], axis=0) 
                                avg_reward_FL_m2,first_ego_FL_m2,loss_FL_m2 = computation(first_ego_FL_m2,avg_reward_FL_m2,
                                                                                              avg_grnd_rwd_FL_m2,deep_max_FL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_m2',write_true)
                                write_true = False
                            #print('FL_m2:')
                            elif (ID_FR and not (ID_F or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_FR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_m2], axis=0) 
                                avg_reward_FR_m2,first_ego_FR_m2,loss_FR_m2 = computation(first_ego_FR_m2,avg_reward_FR_m2,
                                                                                              avg_grnd_rwd_FR_m2,deep_max_FR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_m2',write_true)
                                write_true = False
                            elif (ID_F and not (ID_FR or ID_F2 or ID_B or ID_B2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_m2 = np.mean([ground_r, avg_grnd_rwd_F1_m2], axis=0) 
                                avg_reward_F1_m2,first_ego_F1_m2,loss_F1_m2 = computation(first_ego_F1_m2,avg_reward_F1_m2,
                                                                                              avg_grnd_rwd_F1_m2,deep_max_F1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_m2',write_true)
                                write_true = False
##################################################################################################################
                            elif ((ID_B and ID_BR and ID_BL) and not (ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_BL_m2 = np.mean([ground_r, avg_grnd_rwd_B1_BR_BL_m2], axis=0)
                                avg_reward_B1_BR_BL_m2,first_ego_B1_BR_BL_m2,loss_B1_BR_BL_m2 = computation(first_ego_B1_BR_BL_m2,avg_reward_B1_BR_BL_m2,
                                                                                              avg_grnd_rwd_B1_BR_BL_m2,deep_max_B1_BR_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'B1_BR_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_BL and ID_B2) and not (ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_B2_BR_BL_m2 = np.mean([ground_r, avg_grnd_rwd_B1_B2_BR_BL_m2], axis=0)
                                avg_reward_B1_B2_BR_BL_m2,first_ego_B1_B2_BR_BL_m2,loss_B1_B2_BR_BL_m2 = computation(first_ego_B1_B2_BR_BL_m2,avg_reward_B1_B2_BR_BL_m2,
                                                                                              avg_grnd_rwd_B1_B2_BR_BL_m2,deep_max_B1_B2_BR_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'B1_B2_BR_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BL_B2_m2 = np.mean([ground_r, avg_grnd_rwd_B1_BL_B2_m2], axis=0)  
                                avg_reward_B1_BL_B2_m2,first_ego_B1_BL_B2_m2,loss_B1_BL_B2_m2 = computation(first_ego_B1_BL_B2_m2,avg_reward_B1_BL_B2_m2,
                                                                                              avg_grnd_rwd_B1_BL_B2_m2,deep_max_B1_BL_B2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BL_B2_m2',write_true)
                                write_true = False 
                            elif ((ID_B and ID_B2 and ID_BR) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_B2_m2 = np.mean([ground_r, avg_grnd_rwd_B1_BR_B2_m2], axis=0)  
                                avg_reward_B1_BR_B2_m2,first_ego_B1_BR_B2_m2,loss_B1_BR_B2_m2 = computation(first_ego_B1_BR_B2_m2,avg_reward_B1_BR_B2_m2,
                                                                                              avg_grnd_rwd_B1_BR_B2_m2,deep_max_B1_BR_B2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BR_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2) and not (ID_BL or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_B1_B2_m2], axis=0)  
                                avg_reward_B1_B2_m2,first_ego_B1_B2_m2,loss_B1_B2_m2 = computation(first_ego_B1_B2_m2,avg_reward_B1_B2_m2,
                                                                                              avg_grnd_rwd_B1_B2_m2,deep_max_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR) and not (ID_B2 and ID_BL or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_B1_BR_m2], axis=0)  
                                avg_reward_B1_BR_m2,first_ego_B1_BR_m2,loss_B1_BR_m2 = computation(first_ego_B1_BR_m2,avg_reward_B1_BR_m2,
                                                                                              avg_grnd_rwd_B1_BR_m2,deep_max_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BR__m2',write_true)
                                write_true = False
                            elif ((ID_BL and ID_BR) and not (ID_B2 and ID_B or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_BL_BR_m2], axis=0)  
                                avg_reward_BL_BR_m2,first_ego_BL_BR_m2,loss_BL_BR_m2 = computation(first_ego_BL_BR_m2,avg_reward_BL_BR_m2,
                                                                                              avg_grnd_rwd_BL_BR_m2,deep_max_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'BL_BR__m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BL) and not (ID_B2 and ID_BR or ID_F or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_B1_BL_m2], axis=0)  
                                avg_reward_B1_BL_m2,first_ego_B1_BL_m2,loss_B1_BL_m2 = computation(first_ego_B1_BL_m2,avg_reward_B1_BL_m2,
                                                                                              avg_grnd_rwd_B1_BL_m2,deep_max_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA, 
                                                                                              trajectories,obs_trajectories, 
                                                                                              learning_rate, epochs,'B1_BL_m2',write_true)
                                write_true = False
                            elif (ID_BL and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_BL_m2 = np.mean([ground_r, avg_grnd_rwd_BL_m2], axis=0) 
                                avg_reward_BL_m2,first_ego_BL_m2,loss_BL_m2 = computation(first_ego_BL_m2,avg_reward_BL_m2,
                                                                                              avg_grnd_rwd_BL_m2,deep_max_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'BL_m2',write_true)
                                write_true = False
                            #print('BL_m2:')
                            elif (ID_BR and not (ID_B or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_BR_m2 = np.mean([ground_r, avg_grnd_rwd_BR_m2], axis=0) 
                                avg_reward_BR_m2,first_ego_BR_m2,loss_BR_m2 = computation(first_ego_BR_m2,avg_reward_BR_m2,
                                                                                              avg_grnd_rwd_BR_m2,deep_max_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'BR_m2',write_true)
                                write_true = False
                            elif (ID_B and not (ID_BR or ID_B2 or ID_F or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_B1_m2 = np.mean([ground_r, avg_grnd_rwd_B1_m2], axis=0) 
                                avg_reward_B1_m2,first_ego_B1_m2,loss_B1_m2 = computation(first_ego_B1_m2,avg_reward_B1_m2,
                                                                                              avg_grnd_rwd_B1_m2,deep_max_B1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'B1_m2',write_true)
                                write_true = False
#####################################################################################################################
                            elif ((ID_B and ID_F) and not (ID_BR or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_m2], axis=0)
                                avg_reward_F1_B1_m2,first_ego_F1_B1_m2,loss_F1_B1_m2 = computation(first_ego_F1_B1_m2,avg_reward_F1_B1_m2,
                                                                                              avg_grnd_rwd_F1_B1_m2,deep_max_F1_B1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_F) and not (ID_B or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_BR_m2], axis=0)
                                avg_reward_F1_BR_m2,first_ego_F1_BR_m2,loss_F1_BR_m2 = computation(first_ego_F1_BR_m2,avg_reward_F1_BR_m2,
                                                                                              avg_grnd_rwd_F1_BR_m2,deep_max_F1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BL and ID_F) and not (ID_B or ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_BL_m2], axis=0)
                                avg_reward_F1_BL_m2,first_ego_F1_BL_m2,loss_F1_BL_m2 = computation(first_ego_F1_BL_m2,avg_reward_F1_BL_m2,
                                                                                              avg_grnd_rwd_F1_BL_m2,deep_max_F1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BL_m2',write_true)
                                write_true = False
                            elif (ID_B and ID_BR and ID_B2 and ID_F and ID_F2 and ID_FL and ID_FR and ID_BL):
                                avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m2], axis=0)
                                avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m2,first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m2,loss_F1_F2_FR_FL_B1_B2_BR_BL_m2 = computation(first_ego_F1_F2_FR_FL_B1_B2_BR_BL_m2,
                                                                                                                                                         avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m2,
                                                                                              avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m2,deep_max_F1_F2_FR_FL_B1_B2_BR_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_FL_B1_B2_BR_BL_m2',write_true)
                            elif ((ID_B and ID_F and ID_B2) and not (ID_BR or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_m2], axis=0)
                                avg_reward_F1_B1_B2_m2,first_ego_F1_B1_B2_m2,loss_F1_B1_B2_m2 = computation(first_ego_F1_B1_B2_m2,avg_reward_F1_B1_B2_m2,
                                                                                              avg_grnd_rwd_F1_B1_B2_m2,deep_max_F1_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BR) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BR_m2], axis=0)
                                avg_reward_F1_B1_BR_m2,first_ego_F1_B1_BR_m2,loss_F1_B1_BR_m2 = computation(first_ego_F1_B1_BR_m2,avg_reward_F1_B1_BR_m2,
                                                                                              avg_grnd_rwd_F1_B1_BR_m2,deep_max_F1_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BL_m2], axis=0)
                                avg_reward_F1_B1_BL_m2,first_ego_F1_B1_BL_m2,loss_F1_B1_BL_m2 = computation(first_ego_F1_B1_BL_m2,avg_reward_F1_B1_BL_m2,
                                                                                              avg_grnd_rwd_F1_B1_BL_m2,deep_max_F1_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR or ID_B)):
                                avg_grnd_rwd_F1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_BL_BR_m2], axis=0)
                                avg_reward_F1_BL_BR_m2,first_ego_F1_BL_BR_m2,loss_F1_BL_BR_m2 = computation(first_ego_F1_BL_BR_m2,avg_reward_F1_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_BL_BR_m2,deep_max_F1_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_F and ID_BL) and not (ID_B2 or ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_B1_BL_BR_m2,first_ego_F1_B1_BL_BR_m2,loss_F1_B1_BL_BR_m2 = computation(first_ego_F1_B1_BL_BR_m2,avg_reward_F1_B1_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_B1_BL_BR_m2,deep_max_F1_B1_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_F and ID_BL) and not (ID_F2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_F1_B1_B2_BL_BR_m2,first_ego_F1_B1_B2_BL_BR_m2,loss_F1_B1_B2_BL_BR_m2 = computation(first_ego_F1_B1_B2_BL_BR_m2,avg_reward_F1_B1_B2_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_B1_B2_BL_BR_m2,deep_max_F1_B1_B2_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_B1_B2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B2 and ID_B and ID_F and ID_BL) and not (ID_F2 or ID_FL or ID_FR and ID_BR)):
                                avg_grnd_rwd_F1_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_B1_B2_BL_m2], axis=0)
                                avg_reward_F1_B1_B2_BL_m2,first_ego_F1_B1_B2_BL_m2,loss_F1_B1_B2_BL_m2 = computation(first_ego_F1_B1_B2_BL_m2,avg_reward_F1_B1_B2_BL_m2,avg_grnd_rwd_F1_B1_B2_BL_m2,deep_max_F1_B1_B2_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL and ID_FR) and not (ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BL_m2,first_ego_F1_F2_FL_FR_B1_BL_m2,loss_F1_F2_FL_FR_B1_BL_m2 = computation(first_ego_F1_F2_FL_FR_B1_BL_m2,avg_reward_F1_F2_FL_FR_B1_BL_m2,avg_grnd_rwd_F1_F2_FL_FR_B1_BL_m2,deep_max_F1_F2_FL_FR_B1_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_BL and ID_FR) and not (ID_F2 or ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_BL_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_BL_m2,first_ego_F1_FL_FR_B1_BL_m2,loss_F1_FL_FR_B1_BL_m2 = computation(first_ego_F1_FL_FR_B1_BL_m2,avg_reward_F1_FL_FR_B1_BL_m2,avg_grnd_rwd_F1_FL_FR_B1_BL_m2,deep_max_F1_FL_FR_B1_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_BL and ID_FR and ID_B2) and not (ID_F2 or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BL_m2,first_ego_F1_FL_FR_B1_B2_BL_m2,loss_F1_FL_FR_B1_B2_BL_m2 = computation(first_ego_F1_FL_FR_B1_B2_BL_m2,avg_reward_F1_FL_FR_B1_B2_BL_m2,avg_grnd_rwd_F1_FL_FR_B1_B2_BL_m2,deep_max_F1_FL_FR_B1_B2_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_B2_BL_m2',write_true)
                                write_true = False
################################################################################## F1_F2_combination
                            elif ((ID_B and ID_F and ID_F2) and not (ID_BR or ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_m2], axis=0)
                                avg_reward_F1_F2_B1_m2,first_ego_F1_F2_B1_m2,loss_F1_F2_B1_m2 = computation(first_ego_F1_F2_B1_m2,avg_reward_F1_F2_B1_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_m2,deep_max_F1_F2_B1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_F and ID_F2) and not (ID_B or ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BR_m2], axis=0)
                                avg_reward_F1_F2_BR_m2,first_ego_F1_F2_BR_m2,loss_F1_F2_BR_m2 = computation(first_ego_F1_F2_BR_m2,avg_reward_F1_F2_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_BR_m2,deep_max_F1_F2_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BL and ID_F and ID_F2) and not (ID_B or ID_B2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BL_m2], axis=0)
                                avg_reward_F1_F2_BL_m2,first_ego_F1_F2_BL_m2,loss_F1_F2_BL_m2 = computation(first_ego_F1_F2_BL_m2,avg_reward_F1_F2_BL_m2,
                                                                                              avg_grnd_rwd_F1_F2_BL_m2,deep_max_F1_F2_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_B2 and ID_F2) and not (ID_BR or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_m2], axis=0)
                                avg_reward_F1_F2_B1_B2_m2,first_ego_F1_F2_B1_B2_m2,loss_F1_F2_B1_B2_m2 = computation(first_ego_F1_F2_B1_B2_m2,avg_reward_F1_F2_B1_B2_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_m2,deep_max_F1_F2_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BR and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BR_m2], axis=0)
                                avg_reward_F1_F2_B1_BR_m2,first_ego_F1_F2_B1_BR_m2,loss_F1_F2_B1_BR_m2 = computation(first_ego_F1_F2_B1_BR_m2,avg_reward_F1_F2_B1_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_BR_m2,deep_max_F1_F2_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BL_m2], axis=0)
                                avg_reward_F1_F2_B1_BL_m2,first_ego_F1_F2_B1_BL_m2,loss_F1_F2_B1_BL_m2 = computation(first_ego_F1_F2_B1_BL_m2,avg_reward_F1_F2_B1_BL_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_BL_m2,deep_max_F1_F2_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BL_m2',write_true)
                                write_true = False

                            elif ((ID_BR and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR or ID_B)):
                                avg_grnd_rwd_F1_F2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_BL_BR_m2,first_ego_F1_F2_BL_BR_m2,loss_F1_F2_BL_BR_m2 = computation(first_ego_F1_F2_BL_BR_m2,avg_reward_F1_F2_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_BL_BR_m2,deep_max_F1_F2_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_F and ID_BL and ID_F2) and not (ID_B2 or ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_F2_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_B1_BL_BR_m2,first_ego_F1_F2_B1_BL_BR_m2,loss_F1_F2_B1_BL_BR_m2 = computation(first_ego_F1_F2_B1_BL_BR_m2,avg_reward_F1_F2_B1_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_BL_BR_m2,deep_max_F1_F2_B1_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_F and ID_BL and ID_F2) and not (ID_FL or ID_FR)):
                                avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_B1_B2_BL_BR_m2,first_ego_F1_F2_B1_B2_BL_BR_m2,loss_F1_F2_B1_B2_BL_BR_m2 = computation(first_ego_F1_F2_B1_B2_BL_BR_m2,avg_reward_F1_F2_B1_B2_BL_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_BL_BR_m2,deep_max_F1_F2_B1_B2_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_BL_BR_m2',write_true)
                                write_true = False
############################################################################################################
                            elif ((ID_B and ID_FR) and not (ID_BR or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_m2], axis=0)
                                avg_reward_FR_B1_m2,first_ego_FR_B1_m2,loss_FR_B1_m2 = computation(first_ego_FR_B1_m2,avg_reward_FR_B1_m2,
                                                                                              avg_grnd_rwd_FR_B1_m2,deep_max_FR_B1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_FR) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_BR_m2], axis=0)
                                avg_reward_FR_BR_m2,first_ego_FR_BR_m2,loss_FR_BR_m2 = computation(first_ego_FR_BR_m2,avg_reward_FR_BR_m2,
                                                                                              avg_grnd_rwd_FR_BR_m2,deep_max_FR_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BL and ID_FR) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FL or ID_BR)):
                                avg_grnd_rwd_FR_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FR_BL_m2], axis=0)
                                avg_reward_FR_BL_m2,first_ego_FR_BL_m2,loss_FR_BL_m2 = computation(first_ego_FR_BL_m2,avg_reward_FR_BL_m2,
                                                                                              avg_grnd_rwd_FR_BL_m2,deep_max_FR_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_B2) and not (ID_BR or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_m2], axis=0)
                                avg_reward_FR_B1_B2_m2,first_ego_FR_B1_B2_m2,loss_FR_B1_B2_m2 = computation(first_ego_FR_B1_B2_m2,avg_reward_FR_B1_B2_m2,
                                                                                              avg_grnd_rwd_FR_B1_B2_m2,deep_max_FR_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_BR) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_BL)):
                                avg_grnd_rwd_FR_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BR_m2], axis=0)
                                avg_reward_FR_B1_BR_m2,first_ego_FR_B1_BR_m2,loss_FR_B1_BR_m2 = computation(first_ego_FR_B1_BR_m2,avg_reward_FR_B1_BR_m2,
                                                                                              avg_grnd_rwd_FR_B1_BR_m2,deep_max_FR_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_BR)):
                                avg_grnd_rwd_FR_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BL_m2], axis=0)
                                avg_reward_FR_B1_BL_m2,first_ego_FR_B1_BL_m2,loss_FR_B1_BL_m2 = computation(first_ego_FR_B1_BL_m2,avg_reward_FR_B1_BL_m2,
                                                                                              avg_grnd_rwd_FR_B1_BL_m2,deep_max_FR_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BL_m2',write_true)
                                write_true = False

                            elif ((ID_BR and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL or ID_B)):
                                avg_grnd_rwd_FR_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_BL_BR_m2], axis=0)
                                avg_reward_FR_BL_BR_m2,first_ego_FR_BL_BR_m2,loss_FR_BL_BR_m2 = computation(first_ego_FR_BL_BR_m2,avg_reward_FR_BL_BR_m2,
                                                                                              avg_grnd_rwd_FR_BL_BR_m2,deep_max_FR_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_FR and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FL)):
                                avg_grnd_rwd_FR_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_BL_BR_m2], axis=0)
                                avg_reward_FR_B1_BL_BR_m2,first_ego_FR_B1_BL_BR_m2,loss_FR_B1_BL_BR_m2 = computation(first_ego_FR_B1_BL_BR_m2,avg_reward_FR_B1_BL_BR_m2,
                                                                                              avg_grnd_rwd_FR_B1_BL_BR_m2,deep_max_FR_B1_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_FR and ID_BL) and not (ID_F2 or ID_F or ID_FL)):
                                avg_grnd_rwd_FR_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_FR_B1_B2_BL_BR_m2,first_ego_FR_B1_B2_BL_BR_m2,loss_FR_B1_B2_BL_BR_m2 = computation(first_ego_FR_B1_B2_BL_BR_m2,avg_reward_FR_B1_B2_BL_BR_m2,
                                                                                              avg_grnd_rwd_FR_B1_B2_BL_BR_m2,deep_max_FR_B1_B2_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_BL_BR_m2',write_true)
                                write_true = False
########################################################################################################### FL_combination
                            elif ((ID_B and ID_FL) and not (ID_BR or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_m2], axis=0)
                                avg_reward_FL_B1_m2,first_ego_FL_B1_m2,loss_FL_B1_m2 = computation(first_ego_FL_B1_m2,avg_reward_FL_B1_m2,
                                                                                              avg_grnd_rwd_FL_B1_m2,deep_max_FL_B1_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_FL) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_BR_m2], axis=0)
                                avg_reward_FL_BR_m2,first_ego_FL_BR_m2,loss_FL_BR_m2 = computation(first_ego_FL_BR_m2,avg_reward_FL_BR_m2,
                                                                                              avg_grnd_rwd_FL_BR_m2,deep_max_FL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BL and ID_FL) and not (ID_B or ID_B2 or ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FL_BL_m2], axis=0)
                                avg_reward_FL_BL_m2,first_ego_FL_BL_m2,loss_FL_BL_m2 = computation(first_ego_FL_BL_m2,avg_reward_FL_BL_m2,
                                                                                              avg_grnd_rwd_FL_BL_m2,deep_max_FL_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_B2) and not (ID_BR or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_m2], axis=0)
                                avg_reward_FL_B1_B2_m2,first_ego_FL_B1_B2_m2,loss_FL_B1_B2_m2 = computation(first_ego_FL_B1_B2_m2,avg_reward_FL_B1_B2_m2,
                                                                                              avg_grnd_rwd_FL_B1_B2_m2,deep_max_FL_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_BR) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_BL)):
                                avg_grnd_rwd_FL_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BR_m2], axis=0)
                                avg_reward_FL_B1_BR_m2,first_ego_FL_B1_BR_m2,loss_FL_B1_BR_m2 = computation(first_ego_FL_B1_BR_m2,avg_reward_FL_B1_BR_m2,
                                                                                              avg_grnd_rwd_FL_B1_BR_m2,deep_max_FL_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BL_m2], axis=0)
                                avg_reward_FL_B1_BL_m2,first_ego_FL_B1_BL_m2,loss_FL_B1_BL_m2 = computation(first_ego_FL_B1_BL_m2,avg_reward_FL_B1_BL_m2,
                                                                                              avg_grnd_rwd_FL_B1_BL_m2,deep_max_FL_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BL_m2',write_true)
                                write_true = False

                            elif ((ID_BR and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR or ID_B)):
                                avg_grnd_rwd_FL_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_BL_BR_m2], axis=0)
                                avg_reward_FL_BL_BR_m2,first_ego_FL_BL_BR_m2,loss_FL_BL_BR_m2 = computation(first_ego_FL_BL_BR_m2,avg_reward_FL_BL_BR_m2,
                                                                                              avg_grnd_rwd_FL_BL_BR_m2,deep_max_FL_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_BL_BR_m2',write_true)
                                write_true = False

                            elif ((ID_F and ID_F2 and ID_FL and ID_BL and ID_BR) and not (ID_B or ID_B2 or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_BL_BR_m2,first_ego_F1_F2_FL_BL_BR_m2,loss_F1_F2_FL_BL_BR_m2 = computation(first_ego_F1_F2_FL_BL_BR_m2,avg_reward_F1_F2_FL_BL_BR_m2,avg_grnd_rwd_F1_F2_FL_BL_BR_m2,deep_max_F1_F2_FL_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BL and ID_BR and ID_FR) and not (ID_B or ID_B2)):
                                avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_BL_BR_m2,first_ego_F1_F2_FL_FR_BL_BR_m2,loss_F1_F2_FL_FR_BL_BR_m2 = computation(first_ego_F1_F2_FL_FR_BL_BR_m2,avg_reward_F1_F2_FL_FR_BL_BR_m2,avg_grnd_rwd_F1_F2_FL_FR_BL_BR_m2,deep_max_F1_F2_FL_FR_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_FL and ID_BL and ID_BR and ID_FR and ID_B and ID_B2 ) and not (ID_F or ID_F2 )):
                                avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_FL_FR_B1_B2_BL_BR_m2,first_ego_FL_FR_B1_B2_BL_BR_m2,loss_FL_FR_B1_B2_BL_BR_m2 = computation(first_ego_FL_FR_B1_B2_BL_BR_m2,avg_reward_FL_FR_B1_B2_BL_BR_m2,avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2,deep_max_FL_FR_B1_B2_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'FL_FR_B1_B2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B and ID_FL and ID_BL) and not (ID_B2 or ID_F2 or ID_F or ID_FR)):
                                avg_grnd_rwd_FL_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_BL_BR_m2], axis=0)
                                avg_reward_FL_B1_BL_BR_m2,first_ego_FL_B1_BL_BR_m2,loss_FL_B1_BL_BR_m2 = computation(first_ego_FL_B1_BL_BR_m2,avg_reward_FL_B1_BL_BR_m2,
                                                                                              avg_grnd_rwd_FL_B1_BL_BR_m2,deep_max_FL_B1_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_BR and ID_B2 and ID_B and ID_FL and ID_BL) and not (ID_F2 or ID_F or ID_FR)):
                                avg_grnd_rwd_FL_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_FL_B1_B2_BL_BR_m2,first_ego_FL_B1_B2_BL_BR_m2,loss_FL_B1_B2_BL_BR_m2 = computation(first_ego_FL_B1_B2_BL_BR_m2,avg_reward_FL_B1_B2_BL_BR_m2,
                                                                                              avg_grnd_rwd_FL_B1_B2_BL_BR_m2,deep_max_FL_B1_B2_BL_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_BL_BR_m2',write_true)
                                write_true = False
#######################################################################################################################
                            elif ((ID_F and ID_F2 and ID_FL and ID_BR) and not (ID_B or ID_B2 or ID_FR or ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_BR_m2,first_ego_F1_F2_FL_BR_m2,loss_F1_F2_FL_BR_m2 = computation(first_ego_F1_F2_FL_BR_m2,avg_reward_F1_F2_FL_BR_m2,avg_grnd_rwd_F1_F2_FL_BR_m2,deep_max_F1_F2_FL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR) and not (ID_B2 or ID_BL or ID_FR )):
                                avg_grnd_rwd_F1_F2_FL_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_B1_BR_m2,first_ego_F1_F2_FL_B1_BR_m2,loss_F1_F2_FL_B1_BR_m2 = computation(first_ego_F1_F2_FL_B1_BR_m2,avg_reward_F1_F2_FL_B1_BR_m2,avg_grnd_rwd_F1_F2_FL_B1_BR_m2,deep_max_F1_F2_FL_B1_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B) and not (ID_B2 or ID_BL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_B1_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_m2], axis=0)
                                avg_reward_F1_F2_FL_B1_m2,first_ego_F1_F2_FL_B1_m2,loss_F1_F2_FL_B1_m2 = computation(first_ego_F1_F2_FL_B1_m2,avg_reward_F1_F2_FL_B1_m2,avg_grnd_rwd_F1_F2_FL_B1_m2,deep_max_F1_F2_FL_B1_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B) and not (ID_B2 or ID_BL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_B1_m1 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_m1], axis=0)
                                avg_reward_F1_F2_FL_B1_m1,first_ego_F1_F2_FL_B1_m1,loss_F1_F2_FL_B1_m1 = computation(first_ego_F1_F2_FL_B1_m1,avg_reward_F1_F2_FL_B1_m1,avg_grnd_rwd_F1_F2_FL_B1_m1,deep_max_F1_F2_FL_B1_m1,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_m1',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR) and not (ID_B2 or ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BR_m2,first_ego_F1_F2_FL_FR_B1_BR_m2,loss_F1_F2_FL_FR_B1_BR_m2 = computation(first_ego_F1_F2_FL_FR_B1_BR_m2,avg_reward_F1_F2_FL_FR_B1_BR_m2,avg_grnd_rwd_F1_F2_FL_FR_B1_BR_m2,deep_max_F1_F2_FL_FR_B1_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR and ID_B2) and not (ID_BL)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_B2_BR_m2,first_ego_F1_F2_FL_FR_B1_B2_BR_m2,loss_F1_F2_FL_FR_B1_B2_BR_m2 = computation(first_ego_F1_F2_FL_FR_B1_B2_BR_m2,avg_reward_F1_F2_FL_FR_B1_B2_BR_m2,avg_grnd_rwd_F1_F2_FL_FR_B1_B2_BR_m2,deep_max_F1_F2_FL_FR_B1_B2_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_B2_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BR and ID_FR and ID_BL) and not (ID_B2)):
                                avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_B1_BL_BR_m2,first_ego_F1_F2_FL_FR_B1_BL_BR_m2,loss_F1_F2_FL_FR_B1_BL_BR_m2 = computation(first_ego_F1_F2_FL_FR_B1_BL_BR_m2,avg_reward_F1_F2_FL_FR_B1_BL_BR_m2,avg_grnd_rwd_F1_F2_FL_FR_B1_BL_BR_m2,deep_max_F1_F2_FL_FR_B1_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B2 and ID_B and ID_FL and ID_BL) and not (ID_F2 or ID_F or ID_FR or ID_BR)):
                                avg_grnd_rwd_FL_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FL_B1_B2_BL_m2], axis=0)
                                avg_reward_FL_B1_B2_BL_m2,first_ego_FL_B1_B2_BL_m2,loss_FL_B1_B2_BL_m2 = computation(first_ego_FL_B1_B2_BL_m2,avg_reward_FL_B1_B2_BL_m2,
                                                                                              avg_grnd_rwd_FL_B1_B2_BL_m2,deep_max_FL_B1_B2_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FL_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_BL or ID_BR)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_m2,first_ego_F1_FL_FR_B1_B2_m2,loss_F1_FL_FR_B1_B2_m2 = computation(first_ego_F1_FL_FR_B1_B2_m2,avg_reward_F1_FL_FR_B1_B2_m2,
                                                                                              avg_grnd_rwd_F1_FL_FR_B1_B2_m2,deep_max_F1_FL_FR_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FL_FR_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BL and ID_F and ID_FR) and not (ID_F2 or ID_B2 or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_FR_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BL_m2], axis=0)
                                avg_reward_F1_FR_B1_BL_m2,first_ego_F1_FR_B1_BL_m2,loss_F1_FR_B1_BL_m2 = computation(first_ego_F1_FR_B1_BL_m2,avg_reward_F1_FR_B1_BL_m2,
                                                                                              avg_grnd_rwd_F1_FR_B1_BL_m2,deep_max_F1_FR_B1_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BL and  ID_B2 and ID_FR) and not (ID_F2 or ID_BL or ID_BR or ID_FL or ID_F)):
                                avg_grnd_rwd_FR_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_FR_B1_B2_BL_m2], axis=0)
                                avg_reward_FR_B1_B2_BL_m2,first_ego_FR_B1_B2_BL_m2,loss_FR_B1_B2_BL_m2 = computation(first_ego_FR_B1_B2_BL_m2,avg_reward_FR_B1_B2_BL_m2,
                                                                                              avg_grnd_rwd_FR_B1_B2_BL_m2,deep_max_FR_B1_B2_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'FR_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m2], axis=0)
                                avg_reward_F1_F2_FL_B1_B2_BL_m2,first_ego_F1_F2_FL_B1_B2_BL_m2,loss_F1_F2_FL_B1_B2_BL_m2 = computation(first_ego_F1_F2_FL_B1_B2_BL_m2,avg_reward_F1_F2_FL_B1_B2_BL_m2,avg_grnd_rwd_F1_F2_FL_B1_B2_BL_m2,deep_max_F1_F2_FL_B1_B2_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_B2 and ID_BL) and not (ID_BR or ID_FR or ID_F2)):
                                avg_grnd_rwd_F1_FL_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_B1_B2_BL_m2], axis=0)
                                avg_reward_F1_FL_B1_B2_BL_m2,first_ego_F1_FL_B1_B2_BL_m2,loss_F1_FL_B1_B2_BL_m2 = computation(first_ego_F1_FL_B1_B2_BL_m2,avg_reward_F1_FL_B1_B2_BL_m2,avg_grnd_rwd_F1_FL_B1_B2_BL_m2,deep_max_F1_FL_B1_B2_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_B2 and ID_F and ID_FR) and not (ID_F2 or ID_BL or ID_BR or ID_FL)):
                                avg_grnd_rwd_F1_FR_B1_B2_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_B2_m2], axis=0)
                                avg_reward_F1_FR_B1_B2_m2,first_ego_F1_FR_B1_B2_m2,loss_F1_FR_B1_B2_m2 = computation(first_ego_F1_FR_B1_B2_m2,avg_reward_F1_FR_B1_B2_m2,
                                                                                              avg_grnd_rwd_F1_FR_B1_B2_m2,deep_max_F1_FR_B1_B2_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_B2_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BL) and not (ID_B or ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_BL_m2], axis=0)
                                avg_reward_F1_F2_FL_BL_m2,first_ego_F1_F2_FL_BL_m2,loss_F1_F2_FL_BL_m2 = computation(first_ego_F1_F2_FL_BL_m2,avg_reward_F1_F2_FL_BL_m2,avg_grnd_rwd_F1_F2_FL_BL_m2,deep_max_F1_F2_FL_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_BL_m2',write_true)
                                write_true = False
###
                            elif ((ID_F and ID_F2 and ID_FR and ID_B and ID_BL and ID_BR) and not (ID_FL or ID_B2)):
                                avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_FR_B1_BL_BR_m2,first_ego_F1_F2_FR_B1_BL_BR_m2,loss_F1_F2_FR_B1_BL_BR_m2 = computation(first_ego_F1_F2_FR_B1_BL_BR_m2,avg_reward_F1_F2_FR_B1_BL_BR_m2,avg_grnd_rwd_F1_F2_FR_B1_BL_BR_m2,deep_max_F1_F2_FR_B1_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FR_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FR and ID_B and ID_BL) and not (ID_FL or ID_B2 or ID_BR)):
                                avg_grnd_rwd_F1_F2_FR_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BL_m2], axis=0)
                                avg_reward_F1_F2_FR_B1_BL_m2,first_ego_F1_F2_FR_B1_BL_m2,loss_F1_F2_FR_B1_BL_m2 = computation(first_ego_F1_F2_FR_B1_BL_m2,avg_reward_F1_F2_FR_B1_BL_m2,avg_grnd_rwd_F1_F2_FR_B1_BL_m2,deep_max_F1_F2_FR_B1_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FR_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR) and not (ID_FL or ID_B2 or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_FR_B1_BL_BR_m2,first_ego_F1_FR_B1_BL_BR_m2,loss_F1_FR_B1_BL_BR_m2 = computation(first_ego_F1_FR_B1_BL_BR_m2,avg_reward_F1_FR_B1_BL_BR_m2,avg_grnd_rwd_F1_FR_B1_BL_BR_m2,deep_max_F1_FR_B1_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FR_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not (ID_FL or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_F1_FR_B1_B2_BL_BR_m2,first_ego_F1_FR_B1_B2_BL_BR_m2,loss_F1_FR_B1_B2_BL_BR_m2 = computation(first_ego_F1_FR_B1_B2_BL_BR_m2,avg_reward_F1_FR_B1_B2_BL_BR_m2,avg_grnd_rwd_F1_FR_B1_B2_BL_BR_m2,deep_max_F1_FR_B1_B2_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FR_B1_B2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_FL and ID_FR and ID_B and ID_BL and ID_BR and ID_B2) and not (ID_F or ID_F2)):
                                avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_FL_FR_B1_B2_BL_BR_m2,first_ego_FL_FR_B1_B2_BL_BR_m2,loss_FL_FR_B1_B2_BL_BR_m2 = computation(first_ego_FL_FR_B1_B2_BL_BR_m2,avg_reward_FL_FR_B1_B2_BL_BR_m2,avg_grnd_rwd_FL_FR_B1_B2_BL_BR_m2,deep_max_FL_FR_B1_B2_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'FL_FR_B1_B2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL and ID_BR) and not (ID_B2 or ID_FR)):
                                avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_B1_BL_BR_m2,first_ego_F1_F2_FL_B1_BL_BR_m2,loss_F1_F2_FL_B1_BL_BR_m2 = computation(first_ego_F1_F2_FL_B1_BL_BR_m2,avg_reward_F1_F2_FL_B1_BL_BR_m2,avg_grnd_rwd_F1_F2_FL_B1_BL_BR_m2,deep_max_F1_F2_FL_B1_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_B and ID_BL) and not (ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_FL_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_B1_BL_m2], axis=0)
                                avg_reward_F1_F2_FL_B1_BL_m2,first_ego_F1_F2_FL_B1_BL_m2,loss_F1_F2_FL_B1_BL_m2 = computation(first_ego_F1_F2_FL_B1_BL_m2,avg_reward_F1_F2_FL_B1_BL_m2,avg_grnd_rwd_F1_F2_FL_B1_BL_m2,deep_max_F1_F2_FL_B1_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_F2) and not (ID_B or ID_B2 or ID_BL or ID_FL)):
                                avg_grnd_rwd_F1_F2_FR_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_BR_m2], axis=0)
                                avg_reward_F1_F2_FR_BR_m2,first_ego_F1_F2_FR_BR_m2,loss_F1_F2_FR_BR_m2 = computation(first_ego_F1_F2_FR_BR_m2 ,avg_reward_F1_F2_FR_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_FR_BR_m2 ,deep_max_F1_F2_FR_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_B and ID_F2) and not (ID_B2 or ID_BL or ID_FL)):
                                avg_grnd_rwd_F1_F2_FR_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FR_B1_BR_m2], axis=0)
                                avg_reward_F1_F2_FR_B1_BR_m2,first_ego_F1_F2_FR_B1_BR_m2,loss_F1_F2_FR_B1_BR_m2 = computation(first_ego_F1_F2_FR_B1_BR_m2 ,avg_reward_F1_F2_FR_B1_BR_m2,
                                                                                              avg_grnd_rwd_F1_F2_FR_B1_BR_m2 ,deep_max_F1_F2_FR_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_FR_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FR and ID_BR and ID_B) and not (ID_B2 or ID_BL or ID_FL or ID_F2)):
                                avg_grnd_rwd_F1_FR_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FR_B1_BR_m2], axis=0)
                                avg_reward_F1_FR_B1_BR_m2,first_ego_F1_FR_B1_BR_m2,loss_F1_FR_B1_BR_m2 = computation(first_ego_F1_FR_B1_BR_m2 ,avg_reward_F1_FR_B1_BR_m2,
                                                                                              avg_grnd_rwd_F1_FR_B1_BR_m2 ,deep_max_F1_FR_B1_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FR_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_FL and ID_B and ID_BL) and not (ID_F2 or ID_B2 or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_FL_B1_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_B1_BL_m2], axis=0)
                                avg_reward_F1_FL_B1_BL_m2,first_ego_F1_FL_B1_BL_m2,loss_F1_FL_B1_BL_m2 = computation(first_ego_F1_FL_B1_BL_m2,avg_reward_F1_FL_B1_BL_m2,avg_grnd_rwd_F1_FL_B1_BL_m2,deep_max_F1_FL_B1_BL_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_B1_BL_m2',write_true)
                                write_true = False
                            elif ((ID_F and ID_F2 and ID_FL and ID_BR and ID_FR) and not (ID_B2 or ID_BL or ID_B)):
                                avg_grnd_rwd_F1_F2_FL_FR_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_FL_FR_BR_m2], axis=0)
                                avg_reward_F1_F2_FL_FR_BR_m2,first_ego_F1_F2_FL_FR_BR_m2,loss_F1_F2_FL_FR_BR_m2 = computation(first_ego_F1_F2_FL_FR_BR_m2,avg_reward_F1_F2_FL_FR_BR_m2,avg_grnd_rwd_F1_F2_FL_FR_BR_m2,deep_max_F1_F2_FL_FR_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_F2_FL_FR_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR and ID_BL) and not (ID_F2)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BL_BR_m2,first_ego_F1_FL_FR_B1_B2_BL_BR_m2,loss_F1_FL_FR_B1_B2_BL_BR_m2 = computation(first_ego_F1_FL_FR_B1_B2_BL_BR_m2,                                           avg_reward_F1_FL_FR_B1_B2_BL_BR_m2,avg_grnd_rwd_F1_FL_FR_B1_B2_BL_BR_m2,deep_max_F1_FL_FR_B1_B2_BL_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_B2_BL_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_B2 or ID_BL)):
                                avg_grnd_rwd_F1_FL_FR_B1_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_BR_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_BR_m2,first_ego_F1_FL_FR_B1_BR_m2,loss_F1_FL_FR_B1_BR_m2 = computation(first_ego_F1_FL_FR_B1_BR_m2,                                           avg_reward_F1_FL_FR_B1_BR_m2,avg_grnd_rwd_F1_FL_FR_B1_BR_m2,deep_max_F1_FL_FR_B1_BR_m2,feature_matrix,gw, GAMMA,trajectories,obs_trajectories,learning_rate, epochs,'F1_FL_FR_B1_BR_m2',write_true)
                                write_true = False
                            elif ((ID_B2 and ID_B and ID_F and ID_BL and ID_F2) and not (ID_FL or ID_FR or ID_BR)):
                                avg_grnd_rwd_F1_F2_B1_B2_BL_m2 = np.mean([ground_r, avg_grnd_rwd_F1_F2_B1_B2_BL_m2], axis=0)
                                avg_reward_F1_F2_B1_B2_BL_m2,first_ego_F1_F2_B1_B2_BL_m2,loss_F1_F2_B1_B2_BL_m2 = computation(first_ego_F1_F2_B1_B2_BL_m2,avg_reward_F1_F2_B1_B2_BL_m2,
                                                                                              avg_grnd_rwd_F1_F2_B1_B2_BL_m2,deep_max_F1_F2_B1_B2_BL_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_F2_B1_B2_BL_m2',write_true)
                                write_true = False
                            elif ((ID_B and ID_BR and ID_B2 and ID_F and ID_FL and ID_FR) and not (ID_F2 or ID_BL)):
                                avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m2 = np.mean([ground_r, avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m2], axis=0)
                                avg_reward_F1_FL_FR_B1_B2_BR_m2,first_ego_F1_FL_FR_B1_B2_BR_m2,loss_F1_FL_FR_B1_B2_BR_m2 = computation(first_ego_F1_FL_FR_B1_B2_BR_m2,avg_reward_F1_FL_FR_B1_B2_BR_m2,
                                                                                              avg_grnd_rwd_F1_FL_FR_B1_B2_BR_m2,deep_max_F1_FL_FR_B1_B2_BR_m2,
                                                                                              feature_matrix,gw, GAMMA,
                                                                                              trajectories,obs_trajectories,
                                                                                              learning_rate, epochs,'F1_FL_FR_B1_B2_BR_m2',write_true)
                                write_true = False
######################################################################################################################
                            else:
                                avg_grnd_rwd_m2 = np.mean([ground_r, avg_grnd_rwd_m2], axis=0) 
                                avg_reward_m2,first_ego_m2,loss_m2 = computation(first_ego_m2,avg_reward_m2,
                                                                                     avg_grnd_rwd_m2,deep_max_m2,
                                                                                     feature_matrix,gw, GAMMA, trajectories,
                                                                                     obs_trajectories, learning_rate, epochs,'m2',write_true)
                                write_true = False
                            #print('m2:')
                        vehicle_found = False
        if (time%100000==0):
            print('time:',time)
            ### if required add the visualizations here
            print('loss_m1:',loss_m1)
            print('loss_m2;',loss_m2)
            print('loss_FL_m1:',loss_FL_m1)
            print('loss_FL_m2:',loss_FL_m2)
            print('loss_FR_m1:',loss_FR_m1)
            print('loss_FR_m2:',loss_FR_m2)
            print('loss_F1_m1:',loss_F1_m1)
            print('loss_F1_m2:',loss_F1_m2)
            print('loss_F1_F2_m1:',loss_F1_F2_m1)
            print('loss_F1_F2_m2:',loss_F1_F2_m2)
            print('loss_F1_FR_m1:',loss_F1_FR_m1)
            print('loss_F1_FR_m2:',loss_F1_FR_m2)
            print('loss_F1_FL_m1:',loss_F1_FL_m1)
            print('loss_F1_FL_m2:',loss_F1_FL_m2)
            print('loss_FL_FR_m1:',loss_FL_FR_m1)
            print('loss_FL_FR_m2:',loss_FL_FR_m2)
            print('loss_F1_FR_F2_m1:',loss_F1_FR_F2_m1)
            print('loss_F1_FR_F2_m2:',loss_F1_FR_F2_m2)
            print('loss_F1_FL_F2_m1:',loss_F1_FL_F2_m1)
            print('loss_F1_FL_F2_m2:',loss_F1_FL_F2_m2)
            print('loss_F1_FR_FL_m1:',loss_F1_FR_FL_m1)
            print('loss_F1_FR_FL_m2:',loss_F1_FR_FL_m2)
            print('loss_F1_F2_FR_FL_m1:',loss_F1_F2_FR_FL_m1)
            print('loss_F1_F2_FR_FL_m2:',loss_F1_F2_FR_FL_m2)
            print('loss_BL_m1:',loss_BL_m1)
            print('loss_BL_m2:',loss_BL_m2)
            print('loss_BR_m1:',loss_BR_m1)
            print('loss_BR_m2:',loss_BR_m2)
            print('loss_B1_m1:',loss_B1_m1)
            print('loss_B1_m2:',loss_B1_m2)
            print('loss_B1_B2_m1:',loss_B1_B2_m1)
            print('loss_B1_B2_m2:',loss_B1_B2_m2)
            print('loss_B1_BR_m1:',loss_B1_BR_m1)
            print('loss_B1_BR_m2:',loss_B1_BR_m2)
            print('loss_B1_BL_m1:',loss_B1_BL_m1)
            print('loss_B1_BL_m2:',loss_B1_BL_m2)
            print('loss_BL_BR_m1:',loss_BL_BR_m1)
            print('loss_BL_BR_m2:',loss_BL_BR_m2)
            print('loss_B1_BR_B2_m1:',loss_B1_BR_B2_m1)
            print('loss_B1_BR_B2_m2:',loss_B1_BR_B2_m2)
            print('loss_B1_BL_B2_m1:',loss_B1_BL_B2_m1)
            print('loss_B1_BL_B2_m2:',loss_B1_BL_B2_m2)
            print('loss_B1_BR_BL_m1:',loss_B1_BR_BL_m1)
            print('loss_B1_BR_BL_m2:',loss_B1_BR_BL_m2)
            print('loss_B1_B2_BR_BL_m1:',loss_B1_B2_BR_BL_m1)
            print('loss_B1_B2_BR_BL_m2:',loss_B1_B2_BR_BL_m2)
            print('loss_F1_F2_FR_FL_B1_B2_BR_BL_m1:',loss_F1_F2_FR_FL_B1_B2_BR_BL_m1)
            print('loss_F1_F2_FR_FL_B1_B2_BR_BL_m2:',loss_F1_F2_FR_FL_B1_B2_BR_BL_m2)
            print('loss_F1_B1_m1:',loss_F1_B1_m1)
            print('loss_F1_B1_m2:',loss_F1_B1_m2)
            print('loss_F1_B1_B2_m1:',loss_F1_B1_B2_m1)
            print('loss_F1_B1_B2_m2:',loss_F1_B1_B2_m2)
            print('loss_F1_B1_BL_m1:',loss_F1_B1_BL_m1)
            print('loss_F1_B1_BL_m2:',loss_F1_B1_BL_m2)
            print('loss_F1_B1_BR_m1:',loss_F1_B1_BR_m1)
            print('loss_F1_B1_BR_m2:',loss_F1_B1_BR_m2)
            print('loss_F1_BL_m1:',loss_F1_BL_m1)
            print('loss_F1_BL_m2:',loss_F1_BL_m2)
            print('loss_F1_BR_m1:',loss_F1_BR_m1)
            print('loss_F1_BR_m2:',loss_F1_BR_m2)
            print('loss_F1_BL_BR_m1:',loss_F1_BL_BR_m1)
            print('loss_F1_BL_BR_m2:',loss_F1_BL_BR_m2)
            print('loss_F1_B1_BL_BR_m1:',loss_F1_B1_BL_BR_m1)
            print('loss_F1_B1_BL_BR_m2:',loss_F1_B1_BL_BR_m2)
            print('loss_F1_B1_B2_BL_BR_m1:',loss_F1_B1_B2_BL_BR_m1)
            print('loss_F1_B1_B2_BL_BR_m2:',loss_F1_B1_B2_BL_BR_m2)
            print('loss_F1_F2_B1_m1:',loss_F1_F2_B1_m1)
            print('loss_F1_F2_B1_m2:',loss_F1_F2_B1_m2)
            print('loss_F1_F2_B1_B2_m1:',loss_F1_F2_B1_B2_m1)
            print('loss_F1_F2_B1_B2_m2:',loss_F1_F2_B1_B2_m2)
            print('loss_F1_F2_B1_BL_m1:',loss_F1_F2_B1_BL_m1)
            print('loss_F1_F2_B1_BL_m2:',loss_F1_F2_B1_BL_m2)
            print('loss_F1_F2_B1_BR_m1:',loss_F1_F2_B1_BR_m1)
            print('loss_F1_F2_B1_BR_m2:',loss_F1_F2_B1_BR_m2)
            print('loss_F1_F2_BL_m1:',loss_F1_F2_BL_m1)
            print('loss_F1_F2_BL_m2:',loss_F1_F2_BL_m2)
            print('loss_F1_F2_BR_m1:',loss_F1_F2_BR_m1)
            print('loss_F1_F2_BR_m2:',loss_F1_F2_BR_m2)
            print('loss_F1_F2_BL_BR_m1:',loss_F1_F2_BL_BR_m1)
            print('loss_F1_F2_BL_BR_m2:',loss_F1_F2_BL_BR_m2)
            print('loss_F1_F2_B1_BL_BR_m1:',loss_F1_F2_B1_BL_BR_m1)
            print('loss_F1_F2_B1_BL_BR_m2:',loss_F1_F2_B1_BL_BR_m2)
            print('loss_F1_F2_B1_B2_BL_BR_m1:',loss_F1_F2_B1_B2_BL_BR_m1)
            print('loss_F1_F2_B1_B2_BL_BR_m2:',loss_F1_F2_B1_B2_BL_BR_m2)
        #computation(time,df,df_in,v_id,gw_m1,gw_m2,trajectories,obs_trajectories,first_ego_m1,first_ego_m2,avg_reward_m1,avg_grnd_rwd_m1,avg_reward_m2,avg_grnd_rwd_m2,deep_max_m1,deep_max_m2)
    #no_obstacle
    
    visualize(avg_grnd_rwd_m1, avg_reward_m1,9,20,'m1',20,7)
    visualize(avg_grnd_rwd_m2, avg_reward_m2,14,20,'m2',30,15)
    #FL_obstacle
    visualize(avg_grnd_rwd_FL_m1, avg_reward_FL_m1,9,20,'FL_m1',20,7)
    visualize(avg_grnd_rwd_FL_m2, avg_reward_FL_m2,14,20,'FL_m2',30,15)
    #FR_obstacle
    visualize(avg_grnd_rwd_FR_m1, avg_reward_FR_m1,9,20,'FR_m1',20,7)
    visualize(avg_grnd_rwd_FR_m2, avg_reward_FR_m2,14,20,'FR_m2',30,15)
    #F1 
    visualize(avg_grnd_rwd_F1_m1, avg_reward_F1_m1,9,20,'F1_m1',20,7)
    visualize(avg_grnd_rwd_F1_m2, avg_reward_F1_m2,14,20,'F1_m2',30,15)
    #F1 and F2
    visualize(avg_grnd_rwd_F1_F2_m1, avg_reward_F1_F2_m1,9,20,'F1_F2_m1',20,7)
    visualize(avg_grnd_rwd_F1_F2_m2, avg_reward_F1_F2_m2,14,20,'FR_m2',30,15)
     #F1 and FR
    visualize(avg_grnd_rwd_F1_FR_m1, avg_reward_F1_FR_m1,9,20,'F1_FR_m1',20,7)
    visualize(avg_grnd_rwd_F1_FR_m2, avg_reward_F1_FR_m2,14,20,'F1_FR_m2',30,15) 
    #F1 and FL
    visualize(avg_grnd_rwd_F1_FL_m1, avg_reward_F1_FL_m1,9,20,'F1_FL_m1',20,7)
    visualize(avg_grnd_rwd_F1_FL_m2, avg_reward_F1_FL_m2,14,20,'F1_FL_m2',30,15)
    #FR and FL
    visualize(avg_grnd_rwd_FL_FR_m1, avg_reward_FL_FR_m1,9,20,'FL_FR_m1',20,7)
    visualize(avg_grnd_rwd_FL_FR_m2, avg_reward_FL_FR_m2,14,20,'FL_FR_m2',30,15)
    
    visualize(avg_grnd_rwd_F1_FR_F2_m1, avg_reward_F1_FR_F2_m1,9,20,'F1_FR_F2_m1',20,7)
    visualize(avg_grnd_rwd_F1_FR_F2_m2, avg_reward_F1_FR_F2_m2,14,20,'F1_FR_F2_m2',30,15)
    
    visualize(avg_grnd_rwd_F1_FL_F2_m1, avg_reward_F1_FL_F2_m1,9,20,'F1_FL_F2_m1',20,7)
    visualize(avg_grnd_rwd_F1_FL_F2_m2, avg_reward_F1_FL_F2_m2,14,20,'F1_FL_F2_m2',30,15)

    visualize(avg_grnd_rwd_F1_FR_FL_m1, avg_reward_F1_FR_FL_m1,9,20,'F1_FR_FL_m1',20,7)
    visualize(avg_grnd_rwd_F1_FR_FL_m2, avg_reward_F1_FR_FL_m2,14,20,'F1_FR_FL_m2',30,15)

    visualize(avg_grnd_rwd_F1_F2_FR_FL_m1, avg_reward_F1_F2_FR_FL_m1,9,20,'F1_F2_FR_FL_m1',20,7)
    visualize(avg_grnd_rwd_F1_F2_FR_FL_m2, avg_reward_F1_F2_FR_FL_m2,14,20,'F1_F2_FR_FL_m2',30,15)

    #BL_obstacle
    visualize(avg_grnd_rwd_BL_m1, avg_reward_BL_m1,9,20,'BL_m1',20,7)
    visualize(avg_grnd_rwd_BL_m2, avg_reward_BL_m2,14,20,'BL_m2',30,15)
    #BR_obstacle
    visualize(avg_grnd_rwd_BR_m1, avg_reward_BR_m1,9,20,'BR_m1',20,7)
    visualize(avg_grnd_rwd_BR_m2, avg_reward_BR_m2,14,20,'BR_m2',30,15)
    #B1 
    visualize(avg_grnd_rwd_B1_m1, avg_reward_B1_m1,9,20,'B1_m1',20,7)
    visualize(avg_grnd_rwd_B1_m2, avg_reward_B1_m2,14,20,'B1_m2',30,15)
    #B1 and B2
    visualize(avg_grnd_rwd_B1_B2_m1, avg_reward_B1_B2_m1,9,20,'B1_B2_m1',20,7)
    visualize(avg_grnd_rwd_B1_B2_m2, avg_reward_B1_B2_m2,14,20,'BR_m2',30,15)
     #B1 and BR
    visualize(avg_grnd_rwd_B1_BR_m1, avg_reward_B1_BR_m1,9,20,'B1_BR_m1',20,7)
    visualize(avg_grnd_rwd_B1_BR_m2, avg_reward_B1_BR_m2,14,20,'B1_BR_m2',30,15) 
    #B1 and BL
    visualize(avg_grnd_rwd_B1_BL_m1, avg_reward_B1_BL_m1,9,20,'B1_BL_m1',20,7)
    visualize(avg_grnd_rwd_B1_BL_m2, avg_reward_B1_BL_m2,14,20,'B1_BL_m2',30,15)
    #BR and BL
    visualize(avg_grnd_rwd_BL_BR_m1, avg_reward_BL_BR_m1,9,20,'BL_BR_m1',20,7)
    visualize(avg_grnd_rwd_BL_BR_m2, avg_reward_BL_BR_m2,14,20,'BL_BR_m2',30,15)
    
    visualize(avg_grnd_rwd_B1_BR_B2_m1, avg_reward_B1_BR_B2_m1,9,20,'B1_BR_B2_m1',20,7)
    visualize(avg_grnd_rwd_B1_BR_B2_m2, avg_reward_B1_BR_B2_m2,14,20,'B1_BR_B2_m2',30,15)
    
    visualize(avg_grnd_rwd_B1_BL_B2_m1, avg_reward_B1_BL_B2_m1,9,20,'B1_BL_B2_m1',20,7)
    visualize(avg_grnd_rwd_B1_BL_B2_m2, avg_reward_B1_BL_B2_m2,14,20,'B1_BL_B2_m2',30,15)

    visualize(avg_grnd_rwd_B1_BR_BL_m1, avg_reward_B1_BR_BL_m1,9,20,'B1_BR_BL_m1',20,7)
    visualize(avg_grnd_rwd_B1_BR_BL_m2, avg_reward_B1_BR_BL_m2,14,20,'B1_BR_BL_m2',30,15)

    visualize(avg_grnd_rwd_B1_B2_BR_BL_m1, avg_reward_B1_B2_BR_BL_m1,9,20,'B1_B2_BR_BL_m1',20,7)
    visualize(avg_grnd_rwd_B1_B2_BR_BL_m2, avg_reward_B1_B2_BR_BL_m2,14,20,'B1_B2_BR_BL_m2',30,15)

    visualize(avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m1, avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m1,9,20,'F1_F2_FR_FL_B1_B2_BR_BL_m1',20,7)
    visualize(avg_grnd_rwd_F1_F2_FR_FL_B1_B2_BR_BL_m2, avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m2,14,20,'F1_F2_FR_FL_B1_B2_BR_BL_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_m1, avg_reward_F1_B1_m1,9,20,'F1_B1_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_m2, avg_reward_F1_B1_m2,14,20,'F1_B1_m2',30,15)

    visualize(avg_grnd_rwd_F1_BL_m1, avg_reward_F1_BL_m1,9,20,'F1_BL_m1',20,7)
    visualize(avg_grnd_rwd_F1_BL_m2, avg_reward_F1_BL_m2,14,20,'F1_BL_m2',30,15)

    visualize(avg_grnd_rwd_F1_BR_m1, avg_reward_F1_BR_m1,9,20,'F1_BR_m1',20,7)
    visualize(avg_grnd_rwd_F1_BR_m2, avg_reward_F1_BR_m2,14,20,'F1_BR_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_B2_m1, avg_reward_F1_B1_B2_m1,9,20,'F1_B1_B2_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_B2_m2, avg_reward_F1_B1_B2_m2,14,20,'F1_B1_B2_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_BL_m1, avg_reward_F1_B1_BL_m1,9,20,'F1_B1_BL_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_BL_m2, avg_reward_F1_B1_BL_m2,14,20,'F1_B1_BL_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_BR_m1, avg_reward_F1_B1_BR_m1,9,20,'F1_B1_BR_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_BR_m2, avg_reward_F1_B1_BR_m2,14,20,'F1_B1_BR_m2',30,15)

    visualize(avg_grnd_rwd_F1_BL_BR_m1, avg_reward_F1_BL_BR_m1,9,20,'F1_BL_BR_m1',20,7)
    visualize(avg_grnd_rwd_F1_BL_BR_m2, avg_reward_F1_BL_BR_m2,14,20,'F1_BL_BR_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_BL_BR_m1, avg_reward_F1_B1_BL_BR_m1,9,20,'F1_B1_BL_BR_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_BL_BR_m2, avg_reward_F1_B1_BL_BR_m2,14,20,'F1_B1_BL_BR_m2',30,15)

    visualize(avg_grnd_rwd_F1_B1_B2_BL_BR_m1, avg_reward_F1_B1_B2_BL_BR_m1,9,20,'F1_B1_B2_BL_BR_m1',20,7)
    visualize(avg_grnd_rwd_F1_B1_B2_BL_BR_m2, avg_reward_F1_B1_B2_BL_BR_m2,14,20,'F1_B1_B2_BL_BR_m2',30,15)
    
    model = pickle.load(open(model_filename, 'rb'))
    maxent_policy_m1 = value_iteration.find_policy(gw_m1.n_states,
                                                    gw_m1.n_actions,
                                                    gw_m1.transition_probability,
                                                    avg_reward_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_m2 = value_iteration.find_policy(gw_m2.n_states,
                                                    gw_m2.n_actions,
                                                    gw_m2.transition_probability,
                                                    avg_reward_m2,
                                                    discount)
    maxent_policy_FL_m1 = value_iteration.find_policy(gw_FL_m1.n_states,
                                                    gw_FL_m1.n_actions,
                                                    gw_FL_m1.transition_probability,
                                                    avg_reward_FL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_m2 = value_iteration.find_policy(gw_FL_m2.n_states,
                                                    gw_FL_m2.n_actions,
                                                    gw_FL_m2.transition_probability,
                                                    avg_reward_FL_m2,
                                                    discount)

    maxent_policy_FR_m1 = value_iteration.find_policy(gw_FR_m1.n_states,
                                                    gw_FR_m1.n_actions,
                                                    gw_FR_m1.transition_probability,
                                                    avg_reward_FR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_m2 = value_iteration.find_policy(gw_FR_m2.n_states,
                                                    gw_FR_m2.n_actions,
                                                    gw_FR_m2.transition_probability,
                                                    avg_reward_FR_m2,
                                                    discount)
    maxent_policy_F1_m1 = value_iteration.find_policy(gw_F1_m1.n_states,
                                                    gw_F1_m1.n_actions,
                                                    gw_F1_m1.transition_probability,
                                                    avg_reward_F1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_m2 = value_iteration.find_policy(gw_F1_m2.n_states,
                                                    gw_F1_m2.n_actions,
                                                    gw_F1_m2.transition_probability,
                                                    avg_reward_F1_m2,
                                                    discount)
    maxent_policy_F1_F2_m1 = value_iteration.find_policy(gw_F1_F2_m1.n_states,
                                                    gw_F1_F2_m1.n_actions,
                                                    gw_F1_F2_m1.transition_probability,
                                                    avg_reward_F1_F2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_m2 = value_iteration.find_policy(gw_F1_F2_m2.n_states,
                                                    gw_F1_F2_m2.n_actions,
                                                    gw_F1_F2_m2.transition_probability,
                                                    avg_reward_F1_F2_m2,
                                                    discount)
    maxent_policy_F1_FR_m1 = value_iteration.find_policy(gw_F1_FR_m1.n_states,
                                                    gw_F1_FR_m1.n_actions,
                                                    gw_F1_FR_m1.transition_probability,
                                                    avg_reward_F1_FR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_m2 = value_iteration.find_policy(gw_F1_FR_m2.n_states,
                                                    gw_F1_FR_m2.n_actions,
                                                    gw_F1_FR_m2.transition_probability,
                                                    avg_reward_F1_FR_m2,
                                                    discount)
    maxent_policy_F1_FL_m1 = value_iteration.find_policy(gw_F1_FL_m1.n_states,
                                                    gw_F1_FL_m1.n_actions,
                                                    gw_F1_FL_m1.transition_probability,
                                                    avg_reward_F1_FL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_m2 = value_iteration.find_policy(gw_F1_FL_m2.n_states,
                                                    gw_F1_FL_m2.n_actions,
                                                    gw_F1_FL_m2.transition_probability,
                                                    avg_reward_F1_FL_m2,
                                                    discount)
    maxent_policy_FL_FR_m1 = value_iteration.find_policy(gw_FL_FR_m1.n_states,
                                                    gw_FL_FR_m1.n_actions,
                                                    gw_FL_FR_m1.transition_probability,
                                                    avg_reward_FL_FR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_FR_m2 = value_iteration.find_policy(gw_FL_FR_m2.n_states,
                                                    gw_FL_FR_m2.n_actions,
                                                    gw_FL_FR_m2.transition_probability,
                                                    avg_reward_FL_FR_m2,
                                                    discount)
    maxent_policy_F1_FR_F2_m1 = value_iteration.find_policy(gw_F1_FR_F2_m1.n_states,
                                                    gw_F1_FR_F2_m1.n_actions,
                                                    gw_F1_FR_F2_m1.transition_probability,
                                                    avg_reward_F1_FR_F2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_F2_m2 = value_iteration.find_policy(gw_F1_FR_F2_m2.n_states,
                                                    gw_F1_FR_F2_m2.n_actions,
                                                    gw_F1_FR_F2_m2.transition_probability,
                                                    avg_reward_F1_FR_F2_m2,
                                                    discount)
    maxent_policy_F1_FL_F2_m1 = value_iteration.find_policy(gw_F1_FL_F2_m1.n_states,
                                                    gw_F1_FL_F2_m1.n_actions,
                                                    gw_F1_FL_F2_m1.transition_probability,
                                                    avg_reward_F1_FL_F2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_F2_m2 = value_iteration.find_policy(gw_F1_FL_F2_m2.n_states,
                                                    gw_F1_FL_F2_m2.n_actions,
                                                    gw_F1_FL_F2_m2.transition_probability,
                                                    avg_reward_F1_FL_F2_m2,
                                                    discount)
    maxent_policy_F1_FR_FL_m1 = value_iteration.find_policy(gw_F1_FR_FL_m1.n_states,
                                                    gw_F1_FR_FL_m1.n_actions,
                                                    gw_F1_FR_FL_m1.transition_probability,
                                                    avg_reward_F1_FR_FL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_FL_m2 = value_iteration.find_policy(gw_F1_FR_FL_m2.n_states,
                                                    gw_F1_FR_FL_m2.n_actions,
                                                    gw_F1_FR_FL_m2.transition_probability,
                                                    avg_reward_F1_FR_FL_m2,
                                                    discount)
    maxent_policy_F1_F2_FR_FL_m1 = value_iteration.find_policy(gw_F1_F2_FR_FL_m1.n_states,
                                                    gw_F1_F2_FR_FL_m1.n_actions,
                                                    gw_F1_F2_FR_FL_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_FL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_FL_m2 = value_iteration.find_policy(gw_F1_F2_FR_FL_m2.n_states,
                                                    gw_F1_F2_FR_FL_m2.n_actions,
                                                    gw_F1_F2_FR_FL_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_FL_m2,
                                                    discount)
    maxent_policy_F1_F2_FR_BR_m1 = value_iteration.find_policy(gw_F1_F2_FR_BR_m1.n_states,
                                                    gw_F1_F2_FR_BR_m1.n_actions,
                                                    gw_F1_F2_FR_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_BR_m2 = value_iteration.find_policy(gw_F1_F2_FR_BR_m2.n_states,
                                                    gw_F1_F2_FR_BR_m2.n_actions,
                                                    gw_F1_F2_FR_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FR_B1_BR_m1 = value_iteration.find_policy(gw_F1_F2_FR_B1_BR_m1.n_states,
                                                    gw_F1_F2_FR_B1_BR_m1.n_actions,
                                                    gw_F1_F2_FR_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_B1_BR_m2 = value_iteration.find_policy(gw_F1_F2_FR_B1_BR_m2.n_states,
                                                    gw_F1_F2_FR_B1_BR_m2.n_actions,
                                                    gw_F1_F2_FR_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BR_m2,
                                                    discount)
    maxent_policy_F1_FR_B1_BR_m1 = value_iteration.find_policy(gw_F1_FR_B1_BR_m1.n_states,
                                                    gw_F1_FR_B1_BR_m1.n_actions,
                                                    gw_F1_FR_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_FR_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_B1_BR_m2 = value_iteration.find_policy(gw_F1_FR_B1_BR_m2.n_states,
                                                    gw_F1_FR_B1_BR_m2.n_actions,
                                                    gw_F1_FR_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_FR_B1_BR_m2,
                                                    discount)
########################################################################################################################
    maxent_policy_BL_m1 = value_iteration.find_policy(gw_BL_m1.n_states,
                                                    gw_BL_m1.n_actions,
                                                    gw_BL_m1.transition_probability,
                                                    avg_reward_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_BL_m2 = value_iteration.find_policy(gw_BL_m2.n_states,
                                                    gw_BL_m2.n_actions,
                                                    gw_BL_m2.transition_probability,
                                                    avg_reward_BL_m2,
                                                    discount)

    maxent_policy_BR_m1 = value_iteration.find_policy(gw_BR_m1.n_states,
                                                    gw_BR_m1.n_actions,
                                                    gw_BR_m1.transition_probability,
                                                    avg_reward_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_BR_m2 = value_iteration.find_policy(gw_BR_m2.n_states,
                                                    gw_BR_m2.n_actions,
                                                    gw_BR_m2.transition_probability,
                                                    avg_reward_BR_m2,
                                                    discount)
    maxent_policy_B1_m1 = value_iteration.find_policy(gw_B1_m1.n_states,
                                                    gw_B1_m1.n_actions,
                                                    gw_B1_m1.transition_probability,
                                                    avg_reward_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_m2 = value_iteration.find_policy(gw_B1_m2.n_states,
                                                    gw_B1_m2.n_actions,
                                                    gw_B1_m2.transition_probability,
                                                    avg_reward_B1_m2,
                                                    discount)
    maxent_policy_B1_B2_m1 = value_iteration.find_policy(gw_B1_B2_m1.n_states,
                                                    gw_B1_B2_m1.n_actions,
                                                    gw_B1_B2_m1.transition_probability,
                                                    avg_reward_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_B2_m2 = value_iteration.find_policy(gw_B1_B2_m2.n_states,
                                                    gw_B1_B2_m2.n_actions,
                                                    gw_B1_B2_m2.transition_probability,
                                                    avg_reward_B1_B2_m2,
                                                    discount)
    maxent_policy_B1_BR_m1 = value_iteration.find_policy(gw_B1_BR_m1.n_states,
                                                    gw_B1_BR_m1.n_actions,
                                                    gw_B1_BR_m1.transition_probability,
                                                    avg_reward_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_BR_m2 = value_iteration.find_policy(gw_B1_BR_m2.n_states,
                                                    gw_B1_BR_m2.n_actions,
                                                    gw_B1_BR_m2.transition_probability,
                                                    avg_reward_B1_BR_m2,
                                                    discount)
    maxent_policy_B1_BL_m1 = value_iteration.find_policy(gw_B1_BL_m1.n_states,
                                                    gw_B1_BL_m1.n_actions,
                                                    gw_B1_BL_m1.transition_probability,
                                                    avg_reward_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_BL_m2 = value_iteration.find_policy(gw_B1_BL_m2.n_states,
                                                    gw_B1_BL_m2.n_actions,
                                                    gw_B1_BL_m2.transition_probability,
                                                    avg_reward_B1_BL_m2,
                                                    discount)
    maxent_policy_BL_BR_m1 = value_iteration.find_policy(gw_BL_BR_m1.n_states,
                                                    gw_BL_BR_m1.n_actions,
                                                    gw_BL_BR_m1.transition_probability,
                                                    avg_reward_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_BL_BR_m2 = value_iteration.find_policy(gw_BL_BR_m2.n_states,
                                                    gw_BL_BR_m2.n_actions,
                                                    gw_BL_BR_m2.transition_probability,
                                                    avg_reward_BL_BR_m2,
                                                    discount)
    maxent_policy_B1_BR_B2_m1 = value_iteration.find_policy(gw_B1_BR_B2_m1.n_states,
                                                    gw_B1_BR_B2_m1.n_actions,
                                                    gw_B1_BR_B2_m1.transition_probability,
                                                    avg_reward_B1_BR_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_BR_B2_m2 = value_iteration.find_policy(gw_B1_BR_B2_m2.n_states,
                                                    gw_B1_BR_B2_m2.n_actions,
                                                    gw_B1_BR_B2_m2.transition_probability,
                                                    avg_reward_B1_BR_B2_m2,
                                                    discount)
    maxent_policy_B1_BL_B2_m1 = value_iteration.find_policy(gw_B1_BL_B2_m1.n_states,
                                                    gw_B1_BL_B2_m1.n_actions,
                                                    gw_B1_BL_B2_m1.transition_probability,
                                                    avg_reward_B1_BL_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_BL_B2_m2 = value_iteration.find_policy(gw_B1_BL_B2_m2.n_states,
                                                    gw_B1_BL_B2_m2.n_actions,
                                                    gw_B1_BL_B2_m2.transition_probability,
                                                    avg_reward_B1_BL_B2_m2,
                                                    discount)
    maxent_policy_B1_BR_BL_m1 = value_iteration.find_policy(gw_B1_BR_BL_m1.n_states,
                                                    gw_B1_BR_BL_m1.n_actions,
                                                    gw_B1_BR_BL_m1.transition_probability,
                                                    avg_reward_B1_BR_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_BR_BL_m2 = value_iteration.find_policy(gw_B1_BR_BL_m2.n_states,
                                                    gw_B1_BR_BL_m2.n_actions,
                                                    gw_B1_BR_BL_m2.transition_probability,
                                                    avg_reward_B1_BR_BL_m2,
                                                    discount)
    maxent_policy_B1_B2_BR_BL_m1 = value_iteration.find_policy(gw_B1_B2_BR_BL_m1.n_states,
                                                    gw_B1_B2_BR_BL_m1.n_actions,
                                                    gw_B1_B2_BR_BL_m1.transition_probability,
                                                    avg_reward_B1_B2_BR_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_B1_B2_BR_BL_m2 = value_iteration.find_policy(gw_B1_B2_BR_BL_m2.n_states,
                                                    gw_B1_B2_BR_BL_m2.n_actions,
                                                    gw_B1_B2_BR_BL_m2.transition_probability,
                                                    avg_reward_B1_B2_BR_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m1 = value_iteration.find_policy(gw_F1_F2_FR_FL_B1_B2_BR_BL_m1.n_states,
                                                    gw_F1_F2_FR_FL_B1_B2_BR_BL_m1.n_actions,
                                                    gw_F1_F2_FR_FL_B1_B2_BR_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m2 = value_iteration.find_policy(gw_F1_F2_FR_FL_B1_B2_BR_BL_m2.n_states,
                                                    gw_F1_F2_FR_FL_B1_B2_BR_BL_m2.n_actions,
                                                    gw_F1_F2_FR_FL_B1_B2_BR_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_FL_B1_B2_BR_BL_m2,
                                                    discount)

    maxent_policy_F1_B1_m1 = value_iteration.find_policy(gw_F1_B1_m1.n_states,
                                                    gw_F1_B1_m1.n_actions,
                                                    gw_F1_B1_m1.transition_probability,
                                                    avg_reward_F1_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_m2 = value_iteration.find_policy(gw_F1_B1_m2.n_states,
                                                    gw_F1_B1_m2.n_actions,
                                                    gw_F1_B1_m2.transition_probability,
                                                    avg_reward_F1_B1_m2,
                                                    discount)
    maxent_policy_F1_BR_m1 = value_iteration.find_policy(gw_F1_BR_m1.n_states,
                                                    gw_F1_BR_m1.n_actions,
                                                    gw_F1_BR_m1.transition_probability,
                                                    avg_reward_F1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_BR_m2 = value_iteration.find_policy(gw_F1_BR_m2.n_states,
                                                    gw_F1_BR_m2.n_actions,
                                                    gw_F1_BR_m2.transition_probability,
                                                    avg_reward_F1_BR_m2,
                                                    discount)
    maxent_policy_F1_BL_m1 = value_iteration.find_policy(gw_F1_BL_m1.n_states,
                                                    gw_F1_BL_m1.n_actions,
                                                    gw_F1_BL_m1.transition_probability,
                                                    avg_reward_F1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_BL_m2 = value_iteration.find_policy(gw_F1_BL_m2.n_states,
                                                    gw_F1_BL_m2.n_actions,
                                                    gw_F1_BL_m2.transition_probability,
                                                    avg_reward_F1_BL_m2,
                                                    discount)
    maxent_policy_F1_B1_B2_m1 = value_iteration.find_policy(gw_F1_B1_B2_m1.n_states,
                                                    gw_F1_B1_B2_m1.n_actions,
                                                    gw_F1_B1_B2_m1.transition_probability,
                                                    avg_reward_F1_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_B2_m2 = value_iteration.find_policy(gw_F1_B1_B2_m2.n_states,
                                                    gw_F1_B1_B2_m2.n_actions,
                                                    gw_F1_B1_B2_m2.transition_probability,
                                                    avg_reward_F1_B1_B2_m2,
                                                    discount)
    maxent_policy_F1_B1_BL_m1 = value_iteration.find_policy(gw_F1_B1_BL_m1.n_states,
                                                    gw_F1_B1_BL_m1.n_actions,
                                                    gw_F1_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_BL_m2 = value_iteration.find_policy(gw_F1_B1_BL_m2.n_states,
                                                    gw_F1_B1_BL_m2.n_actions,
                                                    gw_F1_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_B1_BR_m1 = value_iteration.find_policy(gw_F1_B1_BR_m1.n_states,
                                                    gw_F1_B1_BR_m1.n_actions,
                                                    gw_F1_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_BR_m2 = value_iteration.find_policy(gw_F1_B1_BR_m2.n_states,
                                                    gw_F1_B1_BR_m2.n_actions,
                                                    gw_F1_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_B1_BR_m2,
                                                    discount)
    maxent_policy_F1_BL_BR_m1 = value_iteration.find_policy(gw_F1_BL_BR_m1.n_states,
                                                    gw_F1_BL_BR_m1.n_actions,
                                                    gw_F1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_BL_BR_m2 = value_iteration.find_policy(gw_F1_BL_BR_m2.n_states,
                                                    gw_F1_BL_BR_m2.n_actions,
                                                    gw_F1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_B1_BL_BR_m1.n_states,
                                                    gw_F1_B1_BL_BR_m1.n_actions,
                                                    gw_F1_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_B1_BL_BR_m2.n_states,
                                                    gw_F1_B1_BL_BR_m2.n_actions,
                                                    gw_F1_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_F1_B1_B2_BL_BR_m1.n_states,
                                                    gw_F1_B1_B2_BL_BR_m1.n_actions,
                                                    gw_F1_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_F1_B1_B2_BL_BR_m2.n_states,
                                                    gw_F1_B1_B2_BL_BR_m2.n_actions,
                                                    gw_F1_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_B1_B2_BL_m1 = value_iteration.find_policy(gw_F1_B1_B2_BL_m1.n_states,
                                                    gw_F1_B1_B2_BL_m1.n_actions,
                                                    gw_F1_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_F1_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_B1_B2_BL_m2 = value_iteration.find_policy(gw_F1_B1_B2_BL_m2.n_states,
                                                    gw_F1_B1_B2_BL_m2.n_actions,
                                                    gw_F1_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_F1_B1_B2_BL_m2,
                                                    discount)
######################################################################################################FR combinatio
    maxent_policy_FR_B1_m1 = value_iteration.find_policy(gw_FR_B1_m1.n_states,
                                                    gw_FR_B1_m1.n_actions,
                                                    gw_FR_B1_m1.transition_probability,
                                                    avg_reward_FR_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_m2 = value_iteration.find_policy(gw_FR_B1_m2.n_states,
                                                    gw_FR_B1_m2.n_actions,
                                                    gw_FR_B1_m2.transition_probability,
                                                    avg_reward_FR_B1_m2,
                                                    discount)
    maxent_policy_FR_BR_m1 = value_iteration.find_policy(gw_FR_BR_m1.n_states,
                                                    gw_FR_BR_m1.n_actions,
                                                    gw_FR_BR_m1.transition_probability,
                                                    avg_reward_FR_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_BR_m2 = value_iteration.find_policy(gw_FR_BR_m2.n_states,
                                                    gw_FR_BR_m2.n_actions,
                                                    gw_FR_BR_m2.transition_probability,
                                                    avg_reward_FR_BR_m2,
                                                    discount)
    maxent_policy_FR_BL_m1 = value_iteration.find_policy(gw_FR_BL_m1.n_states,
                                                    gw_FR_BL_m1.n_actions,
                                                    gw_FR_BL_m1.transition_probability,
                                                    avg_reward_FR_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_BL_m2 = value_iteration.find_policy(gw_FR_BL_m2.n_states,
                                                    gw_FR_BL_m2.n_actions,
                                                    gw_FR_BL_m2.transition_probability,
                                                    avg_reward_FR_BL_m2,
                                                    discount)
    maxent_policy_FR_B1_B2_m1 = value_iteration.find_policy(gw_FR_B1_B2_m1.n_states,
                                                    gw_FR_B1_B2_m1.n_actions,
                                                    gw_FR_B1_B2_m1.transition_probability,
                                                    avg_reward_FR_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_B2_m2 = value_iteration.find_policy(gw_FR_B1_B2_m2.n_states,
                                                    gw_FR_B1_B2_m2.n_actions,
                                                    gw_FR_B1_B2_m2.transition_probability,
                                                    avg_reward_FR_B1_B2_m2,
                                                    discount)
    maxent_policy_FR_B1_BL_m1 = value_iteration.find_policy(gw_FR_B1_BL_m1.n_states,
                                                    gw_FR_B1_BL_m1.n_actions,
                                                    gw_FR_B1_BL_m1.transition_probability,
                                                    avg_reward_FR_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_BL_m2 = value_iteration.find_policy(gw_FR_B1_BL_m2.n_states,
                                                    gw_FR_B1_BL_m2.n_actions,
                                                    gw_FR_B1_BL_m2.transition_probability,
                                                    avg_reward_FR_B1_BL_m2,
                                                    discount)
    maxent_policy_FR_B1_BR_m1 = value_iteration.find_policy(gw_FR_B1_BR_m1.n_states,
                                                    gw_FR_B1_BR_m1.n_actions,
                                                    gw_FR_B1_BR_m1.transition_probability,
                                                    avg_reward_FR_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_BR_m2 = value_iteration.find_policy(gw_FR_B1_BR_m2.n_states,
                                                    gw_FR_B1_BR_m2.n_actions,
                                                    gw_FR_B1_BR_m2.transition_probability,
                                                    avg_reward_FR_B1_BR_m2,
                                                    discount)
    maxent_policy_FR_BL_BR_m1 = value_iteration.find_policy(gw_FR_BL_BR_m1.n_states,
                                                    gw_FR_BL_BR_m1.n_actions,
                                                    gw_FR_BL_BR_m1.transition_probability,
                                                    avg_reward_FR_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_BL_BR_m2 = value_iteration.find_policy(gw_FR_BL_BR_m2.n_states,
                                                    gw_FR_BL_BR_m2.n_actions,
                                                    gw_FR_BL_BR_m2.transition_probability,
                                                    avg_reward_FR_BL_BR_m2,
                                                    discount)
    maxent_policy_FR_B1_BL_BR_m1 = value_iteration.find_policy(gw_FR_B1_BL_BR_m1.n_states,
                                                    gw_FR_B1_BL_BR_m1.n_actions,
                                                    gw_FR_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_FR_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_BL_BR_m2 = value_iteration.find_policy(gw_FR_B1_BL_BR_m2.n_states,
                                                    gw_FR_B1_BL_BR_m2.n_actions,
                                                    gw_FR_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_FR_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_FR_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_FR_B1_B2_BL_BR_m1.n_states,
                                                    gw_FR_B1_B2_BL_BR_m1.n_actions,
                                                    gw_FR_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_FR_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_FR_B1_B2_BL_BR_m2.n_states,
                                                    gw_FR_B1_B2_BL_BR_m2.n_actions,
                                                    gw_FR_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_FR_B1_B2_BL_BR_m2,
                                                    discount)
#################################################################################################### FL combinatio
    maxent_policy_FL_B1_m1 = value_iteration.find_policy(gw_FL_B1_m1.n_states,
                                                    gw_FL_B1_m1.n_actions,
                                                    gw_FL_B1_m1.transition_probability,
                                                    avg_reward_FL_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_m2 = value_iteration.find_policy(gw_FL_B1_m2.n_states,
                                                    gw_FL_B1_m2.n_actions,
                                                    gw_FL_B1_m2.transition_probability,
                                                    avg_reward_FL_B1_m2,
                                                    discount)
    maxent_policy_FL_BR_m1 = value_iteration.find_policy(gw_FL_BR_m1.n_states,
                                                    gw_FL_BR_m1.n_actions,
                                                    gw_FL_BR_m1.transition_probability,
                                                    avg_reward_FL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_BR_m2 = value_iteration.find_policy(gw_FL_BR_m2.n_states,
                                                    gw_FL_BR_m2.n_actions,
                                                    gw_FL_BR_m2.transition_probability,
                                                    avg_reward_FL_BR_m2,
                                                    discount)
    maxent_policy_FL_BL_m1 = value_iteration.find_policy(gw_FL_BL_m1.n_states,
                                                    gw_FL_BL_m1.n_actions,
                                                    gw_FL_BL_m1.transition_probability,
                                                    avg_reward_FL_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_BL_m2 = value_iteration.find_policy(gw_FL_BL_m2.n_states,
                                                    gw_FL_BL_m2.n_actions,
                                                    gw_FL_BL_m2.transition_probability,
                                                    avg_reward_FL_BL_m2,
                                                    discount)
    maxent_policy_FL_B1_B2_m1 = value_iteration.find_policy(gw_FL_B1_B2_m1.n_states,
                                                    gw_FL_B1_B2_m1.n_actions,
                                                    gw_FL_B1_B2_m1.transition_probability,
                                                    avg_reward_FL_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_B2_m2 = value_iteration.find_policy(gw_FL_B1_B2_m2.n_states,
                                                    gw_FL_B1_B2_m2.n_actions,
                                                    gw_FL_B1_B2_m2.transition_probability,
                                                    avg_reward_FL_B1_B2_m2,
                                                    discount)
    maxent_policy_FL_B1_BL_m1 = value_iteration.find_policy(gw_FL_B1_BL_m1.n_states,
                                                    gw_FL_B1_BL_m1.n_actions,
                                                    gw_FL_B1_BL_m1.transition_probability,
                                                    avg_reward_FL_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_BL_m2 = value_iteration.find_policy(gw_FL_B1_BL_m2.n_states,
                                                    gw_FL_B1_BL_m2.n_actions,
                                                    gw_FL_B1_BL_m2.transition_probability,
                                                    avg_reward_FL_B1_BL_m2,
                                                    discount)
    maxent_policy_FL_B1_BR_m1 = value_iteration.find_policy(gw_FL_B1_BR_m1.n_states,
                                                    gw_FL_B1_BR_m1.n_actions,
                                                    gw_FL_B1_BR_m1.transition_probability,
                                                    avg_reward_FL_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_BR_m2 = value_iteration.find_policy(gw_FL_B1_BR_m2.n_states,
                                                    gw_FL_B1_BR_m2.n_actions,
                                                    gw_FL_B1_BR_m2.transition_probability,
                                                    avg_reward_FL_B1_BR_m2,
                                                    discount)
    maxent_policy_FL_BL_BR_m1 = value_iteration.find_policy(gw_FL_BL_BR_m1.n_states,
                                                    gw_FL_BL_BR_m1.n_actions,
                                                    gw_FL_BL_BR_m1.transition_probability,
                                                    avg_reward_FL_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_BL_BR_m2 = value_iteration.find_policy(gw_FL_BL_BR_m2.n_states,
                                                    gw_FL_BL_BR_m2.n_actions,
                                                    gw_FL_BL_BR_m2.transition_probability,
                                                    avg_reward_FL_BL_BR_m2,
                                                    discount)
    maxent_policy_FL_B1_BL_BR_m1 = value_iteration.find_policy(gw_FL_B1_BL_BR_m1.n_states,
                                                    gw_FL_B1_BL_BR_m1.n_actions,
                                                    gw_FL_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_FL_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_BL_BR_m2 = value_iteration.find_policy(gw_FL_B1_BL_BR_m2.n_states,
                                                    gw_FL_B1_BL_BR_m2.n_actions,
                                                    gw_FL_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_FL_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_FL_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_FL_B1_B2_BL_BR_m1.n_states,
                                                    gw_FL_B1_B2_BL_BR_m1.n_actions,
                                                    gw_FL_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_FL_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_FL_B1_B2_BL_BR_m2.n_states,
                                                    gw_FL_B1_B2_BL_BR_m2.n_actions,
                                                    gw_FL_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_FL_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_FL_B1_B2_BL_m1 = value_iteration.find_policy(gw_FL_B1_B2_BL_m1.n_states,
                                                    gw_FL_B1_B2_BL_m1.n_actions,
                                                    gw_FL_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_FL_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_B1_B2_BL_m2 = value_iteration.find_policy(gw_FL_B1_B2_BL_m2.n_states,
                                                    gw_FL_B1_B2_BL_m2.n_actions,
                                                    gw_FL_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_FL_B1_B2_BL_m2,
                                                    discount)
#####################################################################################################F1_F2 combination
    maxent_policy_F1_F2_B1_m1 = value_iteration.find_policy(gw_F1_F2_B1_m1.n_states,
                                                    gw_F1_F2_B1_m1.n_actions,
                                                    gw_F1_F2_B1_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_m2 = value_iteration.find_policy(gw_F1_F2_B1_m2.n_states,
                                                    gw_F1_F2_B1_m2.n_actions,
                                                    gw_F1_F2_B1_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_m2,
                                                    discount)
    maxent_policy_F1_F2_BR_m1 = value_iteration.find_policy(gw_F1_F2_BR_m1.n_states,
                                                    gw_F1_F2_BR_m1.n_actions,
                                                    gw_F1_F2_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_BR_m2 = value_iteration.find_policy(gw_F1_F2_BR_m2.n_states,
                                                    gw_F1_F2_BR_m2.n_actions,
                                                    gw_F1_F2_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_BL_m1 = value_iteration.find_policy(gw_F1_F2_BL_m1.n_states,
                                                    gw_F1_F2_BL_m1.n_actions,
                                                    gw_F1_F2_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_BL_m2 = value_iteration.find_policy(gw_F1_F2_BL_m2.n_states,
                                                    gw_F1_F2_BL_m2.n_actions,
                                                    gw_F1_F2_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_B2_m1 = value_iteration.find_policy(gw_F1_F2_B1_B2_m1.n_states,
                                                    gw_F1_F2_B1_B2_m1.n_actions,
                                                    gw_F1_F2_B1_B2_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_B2_m2 = value_iteration.find_policy(gw_F1_F2_B1_B2_m2.n_states,
                                                    gw_F1_F2_B1_B2_m2.n_actions,
                                                    gw_F1_F2_B1_B2_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_BL_m1 = value_iteration.find_policy(gw_F1_F2_B1_BL_m1.n_states,
                                                    gw_F1_F2_B1_BL_m1.n_actions,
                                                    gw_F1_F2_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_BL_m2 = value_iteration.find_policy(gw_F1_F2_B1_BL_m2.n_states,
                                                    gw_F1_F2_B1_BL_m2.n_actions,
                                                    gw_F1_F2_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_BR_m1 = value_iteration.find_policy(gw_F1_F2_B1_BR_m1.n_states,
                                                    gw_F1_F2_B1_BR_m1.n_actions,
                                                    gw_F1_F2_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_BR_m2 = value_iteration.find_policy(gw_F1_F2_B1_BR_m2.n_states,
                                                    gw_F1_F2_B1_BR_m2.n_actions,
                                                    gw_F1_F2_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_BL_BR_m1.n_states,
                                                    gw_F1_F2_BL_BR_m1.n_actions,
                                                    gw_F1_F2_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_BL_BR_m2.n_states,
                                                    gw_F1_F2_BL_BR_m2.n_actions,
                                                    gw_F1_F2_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_B1_BL_BR_m1.n_states,
                                                    gw_F1_F2_B1_BL_BR_m1.n_actions,
                                                    gw_F1_F2_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_B1_BL_BR_m2.n_states,
                                                    gw_F1_F2_B1_BL_BR_m2.n_actions,
                                                    gw_F1_F2_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_B1_B2_BL_BR_m1.n_states,
                                                    gw_F1_F2_B1_B2_BL_BR_m1.n_actions,
                                                    gw_F1_F2_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_B1_B2_BL_BR_m2.n_states,
                                                    gw_F1_F2_B1_B2_BL_BR_m2.n_actions,
                                                    gw_F1_F2_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_B1_B2_BL_m1 = value_iteration.find_policy(gw_F1_F2_B1_B2_BL_m1.n_states,
                                                    gw_F1_F2_B1_B2_BL_m1.n_actions,
                                                    gw_F1_F2_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_B1_B2_BL_m2 = value_iteration.find_policy(gw_F1_F2_B1_B2_BL_m2.n_states,
                                                    gw_F1_F2_B1_B2_BL_m2.n_actions,
                                                    gw_F1_F2_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_B1_B2_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_BR_m1.n_states,
                                                    gw_F1_F2_FL_BR_m1.n_actions,
                                                    gw_F1_F2_FL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_BR_m2.n_states,
                                                    gw_F1_F2_FL_BR_m2.n_actions,
                                                    gw_F1_F2_FL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_BL_m1 = value_iteration.find_policy(gw_F1_F2_FL_BL_m1.n_states,
                                                    gw_F1_F2_FL_BL_m1.n_actions,
                                                    gw_F1_F2_FL_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_BL_m2 = value_iteration.find_policy(gw_F1_F2_FL_BL_m2.n_states,
                                                    gw_F1_F2_FL_BL_m2.n_actions,
                                                    gw_F1_F2_FL_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_BL_BR_m1.n_states,
                                                    gw_F1_F2_FL_BL_BR_m1.n_actions,
                                                    gw_F1_F2_FL_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_BL_BR_m2.n_states,
                                                    gw_F1_F2_FL_BL_BR_m2.n_actions,
                                                    gw_F1_F2_FL_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_BL_BR_m1.n_states,
                                                    gw_F1_F2_FL_FR_BL_BR_m1.n_actions,
                                                    gw_F1_F2_FL_FR_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_BL_BR_m2.n_states,
                                                    gw_F1_F2_FL_FR_BL_BR_m2.n_actions,
                                                    gw_F1_F2_FL_FR_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_BL_BR_m2,
                                                    discount)
    maxent_policy_FL_FR_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_FL_FR_B1_B2_BL_BR_m1.n_states,
                                                    gw_FL_FR_B1_B2_BL_BR_m1.n_actions,
                                                    gw_FL_FR_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_FL_FR_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_FR_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_FL_FR_B1_B2_BL_BR_m2.n_states,
                                                    gw_FL_FR_B1_B2_BL_BR_m2.n_actions,
                                                    gw_FL_FR_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_FL_FR_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_FL_FR_B1_B2_BL_m1 = value_iteration.find_policy(gw_FL_FR_B1_B2_BL_m1.n_states,
                                                    gw_FL_FR_B1_B2_BL_m1.n_actions,
                                                    gw_FL_FR_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_FL_FR_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FL_FR_B1_B2_BL_m2 = value_iteration.find_policy(gw_FL_FR_B1_B2_BL_m2.n_states,
                                                    gw_FL_FR_B1_B2_BL_m2.n_actions,
                                                    gw_FL_FR_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_FL_FR_B1_B2_BL_m2,
                                                    discount)
												
    maxent_policy_F1_F2_FL_B1_BL_m1 = value_iteration.find_policy(gw_F1_F2_FL_B1_BL_m1.n_states,
                                                    gw_F1_F2_FL_B1_BL_m1.n_actions,
                                                    gw_F1_F2_FL_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_B1_BL_m2 = value_iteration.find_policy(gw_F1_F2_FL_B1_BL_m2.n_states,
                                                    gw_F1_F2_FL_B1_BL_m2.n_actions,
                                                    gw_F1_F2_FL_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_B1_BL_BR_m1.n_states,
                                                    gw_F1_F2_FL_B1_BL_BR_m1.n_actions,
                                                    gw_F1_F2_FL_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_B1_BL_BR_m2.n_states,
                                                    gw_F1_F2_FL_B1_BL_BR_m2.n_actions,
                                                    gw_F1_F2_FL_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_FL_B1_BL_m1 = value_iteration.find_policy(gw_F1_FL_B1_BL_m1.n_states,
                                                    gw_F1_FL_B1_BL_m1.n_actions,
                                                    gw_F1_FL_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_FL_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_B1_BL_m2 = value_iteration.find_policy(gw_F1_FL_B1_BL_m2.n_states,
                                                    gw_F1_FL_B1_BL_m2.n_actions,
                                                    gw_F1_FL_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_FL_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_B1_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_B1_BR_m1.n_states,
                                                    gw_F1_F2_FL_B1_BR_m1.n_actions,
                                                    gw_F1_F2_FL_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_B1_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_B1_BR_m2.n_states,
                                                    gw_F1_F2_FL_B1_BR_m2.n_actions,
                                                    gw_F1_F2_FL_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_BR_m2,
                                                    discount)
##
    maxent_policy_F1_F2_FR_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FR_B1_BL_BR_m1.n_states,
                                                    gw_F1_F2_FR_B1_BL_BR_m1.n_actions,
                                                    gw_F1_F2_FR_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FR_B1_BL_BR_m2.n_states,
                                                    gw_F1_F2_FR_B1_BL_BR_m2.n_actions,
                                                    gw_F1_F2_FR_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FR_B1_BL_m1 = value_iteration.find_policy(gw_F1_F2_FR_B1_BL_m1.n_states,
                                                    gw_F1_F2_FR_B1_BL_m1.n_actions,
                                                    gw_F1_F2_FR_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FR_B1_BL_m2 = value_iteration.find_policy(gw_F1_F2_FR_B1_BL_m2.n_states,
                                                    gw_F1_F2_FR_B1_BL_m2.n_actions,
                                                    gw_F1_F2_FR_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FR_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_FR_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_FR_B1_BL_BR_m1.n_states,
                                                    gw_F1_FR_B1_BL_BR_m1.n_actions,
                                                    gw_F1_FR_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_FR_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_FR_B1_BL_BR_m2.n_states,
                                                    gw_F1_FR_B1_BL_BR_m2.n_actions,
                                                    gw_F1_FR_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_FR_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_FR_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_F1_FR_B1_B2_BL_BR_m1.n_states,
                                                    gw_F1_FR_B1_B2_BL_BR_m1.n_actions,
                                                    gw_F1_FR_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_FR_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_F1_FR_B1_B2_BL_BR_m2.n_states,
                                                    gw_F1_FR_B1_B2_BL_BR_m2.n_actions,
                                                    gw_F1_FR_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_FR_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_B1_m1 = value_iteration.find_policy(gw_F1_F2_FL_B1_m1.n_states,
                                                    gw_F1_F2_FL_B1_m1.n_actions,
                                                    gw_F1_F2_FL_B1_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_B1_m2 = value_iteration.find_policy(gw_F1_F2_FL_B1_m2.n_states,
                                                    gw_F1_F2_FL_B1_m2.n_actions,
                                                    gw_F1_F2_FL_B1_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_B1_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BR_m1.n_states,
                                                    gw_F1_F2_FL_FR_B1_BR_m1.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_B1_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BR_m2.n_states,
                                                    gw_F1_F2_FL_FR_B1_BR_m2.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_B1_BL_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BL_m1.n_states,
                                                    gw_F1_F2_FL_FR_B1_BL_m1.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_B1_BL_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BL_m2.n_states,
                                                    gw_F1_F2_FL_FR_B1_BL_m2.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_BL_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_BL_m1.n_states,
                                                    gw_F1_FL_FR_B1_BL_m1.n_actions,
                                                    gw_F1_FL_FR_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_BL_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_BL_m2.n_states,
                                                    gw_F1_FL_FR_B1_BL_m2.n_actions,
                                                    gw_F1_FL_FR_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_BL_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_B2_BL_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BL_m1.n_states,
                                                    gw_F1_FL_FR_B1_B2_BL_m1.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_B2_BL_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BL_m2.n_states,
                                                    gw_F1_FL_FR_B1_B2_BL_m2.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BL_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_BR_m1.n_states,
                                                    gw_F1_F2_FL_FR_BR_m1.n_actions,
                                                    gw_F1_F2_FL_FR_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_BR_m2.n_states,
                                                    gw_F1_F2_FL_FR_BR_m2.n_actions,
                                                    gw_F1_F2_FL_FR_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_B1_B2_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_B2_BR_m1.n_states,
                                                    gw_F1_F2_FL_FR_B1_B2_BR_m1.n_actions,
                                                    gw_F1_F2_FL_FR_B1_B2_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_B2_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_B1_B2_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_B2_BR_m2.n_states,
                                                    gw_F1_F2_FL_FR_B1_B2_BR_m2.n_actions,
                                                    gw_F1_F2_FL_FR_B1_B2_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_B2_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_FR_B1_BL_BR_m1 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BL_BR_m1.n_states,
                                                    gw_F1_F2_FL_FR_B1_BL_BR_m1.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_FR_B1_BL_BR_m2 = value_iteration.find_policy(gw_F1_F2_FL_FR_B1_BL_BR_m2.n_states,
                                                    gw_F1_F2_FL_FR_B1_BL_BR_m2.n_actions,
                                                    gw_F1_F2_FL_FR_B1_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_FR_B1_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_F2_FL_B1_B2_BL_m1 = value_iteration.find_policy(gw_F1_F2_FL_B1_B2_BL_m1.n_states,
                                                    gw_F1_F2_FL_B1_B2_BL_m1.n_actions,
                                                    gw_F1_F2_FL_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_F2_FL_B1_B2_BL_m2 = value_iteration.find_policy(gw_F1_F2_FL_B1_B2_BL_m2.n_states,
                                                    gw_F1_F2_FL_B1_B2_BL_m2.n_actions,
                                                    gw_F1_F2_FL_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_F1_F2_FL_B1_B2_BL_m2,
                                                    discount)
    maxent_policy_F1_FL_B1_B2_BL_m1 = value_iteration.find_policy(gw_F1_FL_B1_B2_BL_m1.n_states,
                                                    gw_F1_FL_B1_B2_BL_m1.n_actions,
                                                    gw_F1_FL_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_F1_FL_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_B1_B2_BL_m2 = value_iteration.find_policy(gw_F1_FL_B1_B2_BL_m2.n_states,
                                                    gw_F1_FL_B1_B2_BL_m2.n_actions,
                                                    gw_F1_FL_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_F1_FL_B1_B2_BL_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_B2_BL_BR_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BL_BR_m1.n_states,
                                                    gw_F1_FL_FR_B1_B2_BL_BR_m1.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BL_BR_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BL_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_B2_BL_BR_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BL_BR_m2.n_states,
                                                    gw_F1_FL_FR_B1_B2_BL_BR_m2.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BL_BR_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BL_BR_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_B2_BR_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BR_m1.n_states,
                                                    gw_F1_FL_FR_B1_B2_BR_m1.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BR_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_B2_BR_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_BR_m2.n_states,
                                                    gw_F1_FL_FR_B1_B2_BR_m2.n_actions,
                                                    gw_F1_FL_FR_B1_B2_BR_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_BR_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_B2_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_m1.n_states,
                                                    gw_F1_FL_FR_B1_B2_m1.n_actions,
                                                    gw_F1_FL_FR_B1_B2_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_B2_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_B2_m2.n_states,
                                                    gw_F1_FL_FR_B1_B2_m2.n_actions,
                                                    gw_F1_FL_FR_B1_B2_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_B2_m2,
                                                    discount)
    maxent_policy_F1_FR_B1_B2_m1 = value_iteration.find_policy(gw_F1_FR_B1_B2_m1.n_states,
                                                    gw_F1_FR_B1_B2_m1.n_actions,
                                                    gw_F1_FR_B1_B2_m1.transition_probability,
                                                    avg_reward_F1_FR_B1_B2_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_B1_B2_m2 = value_iteration.find_policy(gw_F1_FR_B1_B2_m2.n_states,
                                                    gw_F1_FR_B1_B2_m2.n_actions,
                                                    gw_F1_FR_B1_B2_m2.transition_probability,
                                                    avg_reward_F1_FR_B1_B2_m2,
                                                    discount)
    maxent_policy_F1_FR_B1_BL_m1 = value_iteration.find_policy(gw_F1_FR_B1_BL_m1.n_states,
                                                    gw_F1_FR_B1_BL_m1.n_actions,
                                                    gw_F1_FR_B1_BL_m1.transition_probability,
                                                    avg_reward_F1_FR_B1_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FR_B1_BL_m2 = value_iteration.find_policy(gw_F1_FR_B1_BL_m2.n_states,
                                                    gw_F1_FR_B1_BL_m2.n_actions,
                                                    gw_F1_FR_B1_BL_m2.transition_probability,
                                                    avg_reward_F1_FR_B1_BL_m2,
                                                    discount)
    maxent_policy_FR_B1_B2_BL_m1 = value_iteration.find_policy(gw_FR_B1_B2_BL_m1.n_states,
                                                    gw_FR_B1_B2_BL_m1.n_actions,
                                                    gw_FR_B1_B2_BL_m1.transition_probability,
                                                    avg_reward_FR_B1_B2_BL_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_FR_B1_B2_BL_m2 = value_iteration.find_policy(gw_FR_B1_B2_BL_m2.n_states,
                                                    gw_FR_B1_B2_BL_m2.n_actions,
                                                    gw_FR_B1_B2_BL_m2.transition_probability,
                                                    avg_reward_FR_B1_B2_BL_m2,
                                                    discount)
    maxent_policy_F1_FL_FR_B1_BR_m1 = value_iteration.find_policy(gw_F1_FL_FR_B1_BR_m1.n_states,
                                                    gw_F1_FL_FR_B1_BR_m1.n_actions,
                                                    gw_F1_FL_FR_B1_BR_m1.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_BR_m1,
                                                    discount)#value_iteration_1.(gw_m1.transition_probability,r.T,GAMMA)
    maxent_policy_F1_FL_FR_B1_BR_m2 = value_iteration.find_policy(gw_F1_FL_FR_B1_BR_m2.n_states,
                                                    gw_F1_FL_FR_B1_BR_m2.n_actions,
                                                    gw_F1_FL_FR_B1_BR_m2.transition_probability,
                                                    avg_reward_F1_FL_FR_B1_BR_m2,
                                                    discount)
    #test_ids=[1,33,61,92]#test_ids#
    #test_ids=test_ids[0:5]
    dist_all = 0
    df_org = df
    for t_id in test_ids:
        dist=0
        df_test=df.loc[df['ID']==t_id]
        df_test=df_test[df_test.x<15]
        if (df_test['Type'].iloc[0]!= 'Car'):
            continue
        n_trajectories = df_test.shape[0]-1
        print('Trajectory length:',n_trajectories)
        length_traj = int(n_trajectories/3)
        obstacle = True
        df_sub = df_test.loc[:,['Lane','Delta_v_F','Dist_F','Delta_v_F2','Dist_F2','Delta_v_FL','Dist_FL','Delta_v_FR','Dist_FR']]
        to_draw_actual_x = np.zeros((length_traj))
        to_draw_actual_y = np.zeros((length_traj))
        to_draw_result_x = np.zeros((length_traj))
        to_draw_result_y = np.zeros((length_traj))
        actual_speed = np.zeros((length_traj))
        mat=df_test.as_matrix()
        speed_t_1=mat[:, [6]]
        speed_t_1 = np.roll(speed_t_1, 1)
        mat[:, [6]]=speed_t_1
        df_sub['speed']=mat[:,[6]]
        #mat = np.delete(mat, (0), axis=0)
        Xnew = df_sub.as_matrix()
        Xnew = min_max_scaler.transform(Xnew)
        Ynew = model.predict(Xnew)
        j=0
        for i in range(0,n_trajectories-2,3):
            to_draw_actual_x[j]=mat[i][4]
            to_draw_actual_y[j]=mat[i][5]
            actual_speed[j]=mat[i][6]
            j=j+1
        current_states=np.column_stack((to_draw_actual_x,to_draw_actual_y))
        #print('current_states.shape:',current_states.shape)
        xmax,ymax=np.amax(current_states, axis=0)
        xmin=0
        ymin=0
        y_scale=(ymax-ymin)
        x_scale=(xmax-xmin)
        gw.update_scales(x_scale,y_scale)
        current_state = int(gw.point_to_int(((to_draw_actual_x[0]),to_draw_actual_y[0])))
        print('vehicle id:',t_id)
        predict(mat,gw,maxent_policy_FL_FR_B1_B2_BL_m1,maxent_policy_FL_FR_B1_B2_BL_m2,maxent_policy_F1_F2_FL_BL_BR_m1,maxent_policy_F1_F2_FL_BL_BR_m2,maxent_policy_F1_F2_FL_FR_BL_BR_m1,maxent_policy_F1_F2_FL_FR_BL_BR_m2,
maxent_policy_FL_FR_B1_B2_BL_BR_m1,maxent_policy_FL_FR_B1_B2_BL_BR_m2,maxent_policy_F1_FR_B1_B2_BL_BR_m1,maxent_policy_F1_FR_B1_B2_BL_BR_m2,
maxent_policy_F1_FR_B1_BL_BR_m1,maxent_policy_F1_FR_B1_BL_BR_m2,
maxent_policy_F1_F2_FR_B1_BL_m1,maxent_policy_F1_F2_FR_B1_BL_m2,
maxent_policy_F1_F2_FR_B1_BL_BR_m1,maxent_policy_F1_F2_FR_B1_BL_BR_m2,
maxent_policy_F1_F2_FL_FR_B1_BL_m1,maxent_policy_F1_F2_FL_FR_B1_BL_m2,maxent_policy_F1_FL_FR_B1_BL_m1,maxent_policy_F1_FL_FR_B1_BL_m2,maxent_policy_F1_FL_FR_B1_B2_BL_m1,maxent_policy_F1_FL_FR_B1_B2_BL_m2,maxent_policy_F1_FR_B1_BR_m1,maxent_policy_F1_FR_B1_BR_m2,maxent_policy_F1_F2_FR_B1_BR_m1,maxent_policy_F1_F2_FR_B1_BR_m2,maxent_policy_F1_F2_FR_BR_m1,maxent_policy_F1_F2_FR_BR_m2,maxent_policy_FR_B1_B2_BL_m1,maxent_policy_FR_B1_B2_BL_m2,maxent_policy_F1_FR_B1_BL_m1,maxent_policy_F1_FR_B1_BL_m2,maxent_policy_F1_FR_B1_B2_m1,maxent_policy_F1_FR_B1_B2_m2,maxent_policy_F1_FL_FR_B1_B2_m1,maxent_policy_F1_FL_FR_B1_B2_m2,maxent_policy_F1_FL_FR_B1_B2_BR_m1,maxent_policy_F1_FL_FR_B1_B2_BR_m2,maxent_policy_F1_FL_FR_B1_BR_m1,maxent_policy_F1_FL_FR_B1_BR_m2,maxent_policy_F1_F2_FL_B1_BL_BR_m1,maxent_policy_F1_F2_FL_B1_BL_BR_m2,maxent_policy_FL_B1_B2_BL_m1,maxent_policy_FL_B1_B2_BL_m2,maxent_policy_F1_F2_B1_B2_BL_m1,maxent_policy_F1_F2_B1_B2_BL_m2,maxent_policy_F1_FL_FR_B1_B2_BL_BR_m1,maxent_policy_F1_FL_FR_B1_B2_BL_BR_m2,maxent_policy_F1_F2_FL_FR_BR_m1,maxent_policy_F1_F2_FL_FR_BR_m2,maxent_policy_F1_F2_FL_BL_m1,maxent_policy_F1_F2_FL_BL_m2,maxent_policy_F1_F2_FL_B1_BL_m1,maxent_policy_F1_F2_FL_B1_BL_m2,maxent_policy_F1_FL_B1_BL_m1,maxent_policy_F1_FL_B1_BL_m2,
maxent_policy_F1_F2_FL_B1_m1,maxent_policy_F1_F2_FL_B1_m2,maxent_policy_F1_B1_B2_BL_m1,maxent_policy_F1_B1_B2_BL_m2,maxent_policy_F1_F2_FL_BR_m1,maxent_policy_F1_F2_FL_BR_m2,maxent_policy_F1_F2_FL_B1_BR_m1,maxent_policy_F1_F2_FL_B1_BR_m2,maxent_policy_F1_F2_FL_FR_B1_BR_m1,maxent_policy_F1_F2_FL_FR_B1_BR_m2,
maxent_policy_F1_F2_FL_FR_B1_B2_BR_m1,maxent_policy_F1_F2_FL_FR_B1_B2_BR_m2,maxent_policy_F1_F2_FL_FR_B1_BL_BR_m1,maxent_policy_F1_F2_FL_FR_B1_BL_BR_m2,maxent_policy_F1_F2_FL_B1_B2_BL_m1,maxent_policy_F1_F2_FL_B1_B2_BL_m2,
maxent_policy_F1_FL_B1_B2_BL_m1,maxent_policy_F1_FL_B1_B2_BL_m2,
maxent_policy_FR_BL_BR_m1,maxent_policy_FR_BL_BR_m2,maxent_policy_FR_B1_BL_BR_m1,maxent_policy_FR_B1_BL_BR_m2,maxent_policy_FR_B1_B2_BL_BR_m1,maxent_policy_FR_B1_B2_BL_BR_m2,
               maxent_policy_FR_BR_m1,maxent_policy_FR_BR_m2,maxent_policy_FR_BL_m1,maxent_policy_FR_BL_m2,
               maxent_policy_FR_B1_BR_m1,maxent_policy_FR_B1_BR_m2,maxent_policy_FR_B1_BL_m1,maxent_policy_FR_B1_BL_m2,
                maxent_policy_FR_B1_m1,maxent_policy_FR_B1_B2_m2,maxent_policy_FR_B1_B2_m1,maxent_policy_FR_B1_m2,
maxent_policy_FL_BL_BR_m1,maxent_policy_FL_BL_BR_m2,maxent_policy_FL_B1_BL_BR_m1,maxent_policy_FL_B1_BL_BR_m2,maxent_policy_FL_B1_B2_BL_BR_m1,maxent_policy_FL_B1_B2_BL_BR_m2,
               maxent_policy_FL_BR_m1,maxent_policy_FL_BR_m2,maxent_policy_FL_BL_m1,maxent_policy_FL_BL_m2,
               maxent_policy_FL_B1_BR_m1,maxent_policy_FL_B1_BR_m2,maxent_policy_FL_B1_BL_m1,maxent_policy_FL_B1_BL_m2,
                maxent_policy_FL_B1_m1,maxent_policy_FL_B1_B2_m2,maxent_policy_FL_B1_B2_m1,maxent_policy_FL_B1_m2,
maxent_policy_F1_F2_BL_BR_m1,maxent_policy_F1_F2_BL_BR_m2,maxent_policy_F1_F2_B1_BL_BR_m1,maxent_policy_F1_F2_B1_BL_BR_m2,maxent_policy_F1_F2_B1_B2_BL_BR_m1,maxent_policy_F1_F2_B1_B2_BL_BR_m2,
               maxent_policy_F1_F2_BR_m1,maxent_policy_F1_F2_BR_m2,maxent_policy_F1_F2_BL_m1,maxent_policy_F1_F2_BL_m2,
               maxent_policy_F1_F2_B1_BR_m1,maxent_policy_F1_F2_B1_BR_m2,maxent_policy_F1_F2_B1_BL_m1,maxent_policy_F1_F2_B1_BL_m2,
                maxent_policy_F1_F2_B1_m1,maxent_policy_F1_F2_B1_B2_m2,maxent_policy_F1_F2_B1_B2_m1,maxent_policy_F1_F2_B1_m2,maxent_policy_F1_BL_BR_m1,maxent_policy_F1_BL_BR_m2,maxent_policy_F1_B1_BL_BR_m1,maxent_policy_F1_B1_BL_BR_m2,maxent_policy_F1_B1_B2_BL_BR_m1,maxent_policy_F1_B1_B2_BL_BR_m2,
               maxent_policy_F1_BR_m1,maxent_policy_F1_BR_m2,maxent_policy_F1_BL_m1,maxent_policy_F1_BL_m2,
               maxent_policy_F1_B1_BR_m1,maxent_policy_F1_B1_BR_m2,maxent_policy_F1_B1_BL_m1,maxent_policy_F1_B1_BL_m2,
                maxent_policy_F1_B1_m1,maxent_policy_F1_B1_B2_m2,maxent_policy_F1_B1_B2_m1,maxent_policy_F1_B1_m2,
                maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m1,maxent_policy_F1_F2_FR_FL_B1_B2_BR_BL_m2,maxent_policy_B1_B2_BR_BL_m1,maxent_policy_B1_B2_BR_BL_m2,
                maxent_policy_B1_BR_BL_m1,maxent_policy_B1_BR_BL_m2, maxent_policy_B1_BL_B2_m1,maxent_policy_B1_BL_B2_m2,
                maxent_policy_B1_BR_B2_m1,maxent_policy_B1_BR_B2_m2,maxent_policy_B1_BL_m1,
                maxent_policy_B1_BL_m2,maxent_policy_BL_BR_m1,maxent_policy_BL_BR_m2,
                maxent_policy_B1_m1,maxent_policy_B1_m2,maxent_policy_B1_B2_m1,maxent_policy_B1_B2_m2,
                maxent_policy_B1_BR_m1,maxent_policy_B1_BR_m2,maxent_policy_BL_m1,
                maxent_policy_BL_m2, maxent_policy_BR_m1,maxent_policy_BR_m2,
                maxent_policy_F1_F2_FR_FL_m1,maxent_policy_F1_F2_FR_FL_m2,
                maxent_policy_F1_FR_FL_m1,maxent_policy_F1_FR_FL_m2, maxent_policy_F1_FL_F2_m1,maxent_policy_F1_FL_F2_m2,
                maxent_policy_F1_FR_F2_m1,maxent_policy_F1_FR_F2_m2,maxent_policy_F1_FL_m1,
                maxent_policy_F1_FL_m2,maxent_policy_FL_FR_m1,maxent_policy_FL_FR_m2,
                maxent_policy_F1_m1,maxent_policy_F1_m2,maxent_policy_F1_F2_m1,maxent_policy_F1_F2_m2,
                maxent_policy_F1_FR_m1,maxent_policy_F1_FR_m2,maxent_policy_m1,maxent_policy_m2,maxent_policy_FL_m1,
                maxent_policy_FL_m2, maxent_policy_FR_m1,maxent_policy_FR_m2,
                current_state, to_draw_result_x, to_draw_result_y, to_draw_actual_x[0], to_draw_actual_y[0],
                length_traj,t_id,Ynew,actual_speed)
        fig = plt.figure(figsize=(5,5))
        plt.xlim([0,15])
        #plt.ylim([17,22])
        plt.xlabel("x-coord [m]", fontsize=20)
        plt.ylabel("y-coord [m]", fontsize=20)
        plt.title('Vehicle Id:'+str(t_id))
        actual_plt = plt.plot(to_draw_actual_x, to_draw_actual_y,"bo",linestyle='-',label='actual traj')
        pred_plt = plt.plot(to_draw_result_x, to_draw_result_y,"r+",linestyle='--',label='predicted traj')
        plt.legend(loc='upper left')
        print("Actual Coordinates:",to_draw_actual_x[0:n_trajectories], to_draw_actual_y[0:n_trajectories])
        print("Predicted Coordinates:",to_draw_result_x[0:n_trajectories], to_draw_result_y[0:n_trajectories])
        plt.savefig(r"C:\Users\ga67zod\Desktop\Final_IRL\test_result\Vehicle"+str(t_id)+".png")
        P=[] 
        Q=[]
        for x in range(length_traj):
           P.append([[],[]])
           Q.append([[],[]]) 
        
        for r in range(0,length_traj):
            P[r][0]=to_draw_actual_x[r]
            P[r][1]=to_draw_actual_y[r]
    
            Q[r][0]=to_draw_result_x[r]
            Q[r][1]=to_draw_result_y[r]
            
        #calculate the sum of absolute distance between each point
            dist+= math.sqrt( ((P[r][0]-Q[r][0])**2)+((P[r][1]-Q[r][1])**2) )
        #print(P)
        #print(Q)
        Curv_Dist=frdist(P,Q) 
        dist_all=dist_all+dist
        print ('Frechet distance =',Curv_Dist)
        dist=0
        Curv_Dist=0
    print('Total Frechet Distance:',dist_all)
    print('different scenarios and list of IDs: ',dict(list_vids))
           
if __name__ == '__main__':
    min_max_scaler = MinMaxScaler()
    speed(min_max_scaler) #0.000001
    main(9,20,0.00000001,5,5,0.000001)#grid_size_x,grid_size_y, discount, n_trajectories, epochs, learning_rate