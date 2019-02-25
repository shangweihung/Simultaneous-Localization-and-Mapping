# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:03:58 2019

@author: Shang-Wei Hung
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from utils import first_shot
from map_utils import bresenham2D
import sys
import cv2
import imageio

def show_lidar(ranges):
   angles = np.arange(-135,135.25,0.25)*np.pi/180.0
   ax = plt.subplot(111, projection='polar')
   ax.plot(angles, ranges)
   ax.set_rmax(10)
   ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
   ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
   ax.grid(True)
   ax.set_title("Lidar scan data", va='bottom')
   plt.show()

def first_scan(ranges):
      angles = np.arange(-135,135.25,0.25)*np.pi/180.0
      ranges = ranges
    
      # take valid indices
      indValid = np.logical_and((ranges < 30),(ranges> 0.1))
      ranges = ranges[indValid]
      angles = angles[indValid]
    
      # init MAP
      MAP = {}
      MAP['res']   = 0.05 #meters
      MAP['xmin']  = -20  #meters
      MAP['ymin']  = -20
      MAP['xmax']  =  20
      MAP['ymax']  =  20 
      MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
      MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
      MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
      
    
      
      # xy position in the sensor frame
      xs0 = ranges*np.cos(angles)
      ys0 = ranges*np.sin(angles)
      
      #plot original lidar points
      fig1 = plt.figure()
      plt.plot(xs0,ys0,'.k')
      plt.show()



def mapping(grid,lidar_scan,now,res,angle):
    valid_scan = np.logical_and(lidar_scan>=0.1,lidar_scan<=30)
    #valid_idx, = np.where(valid_scan==True)
    
    
    empty_odds = np.log(0.9/0.1)
    occuy_odds = np.log(0.9/0.1)
    
    range_valid = lidar_scan[valid_scan]
    theta = angle[valid_scan] + now[2]
    x = range_valid * np.cos(theta)
    y = range_valid * np.sin(theta)
    
    x_cell = (x/res).astype(int)
    y_cell = (y/res).astype(int)  
    e_cell ={}
    o_cell ={}
    for (a,b) in zip(x_cell,y_cell):
        lines = bresenham2D(0,0,a,b).astype(int)
        xx = a + int(now[0]) + grid.shape[0]//2             
        yy = b + int(now[1]) + grid.shape[1]//2
        
        o_cell[(xx,yy)] = True
        for i in range(len(lines[0])-1): # the last col is wall
            e_cell[lines[0, i],lines[1, i]] = True
    
    for k, _ in o_cell.items():
        if 0<= k[0] <grid.shape[0] and 0<= k[1] <grid.shape[1]:
            grid[k[0],k[1]] += occuy_odds
            
    for k, _ in e_cell.items():
        xxx = k[0] + int(now[0]) + grid.shape[0]//2
        yyy = k[1] + int(now[1]) + grid.shape[1]//2
        if 0<= xxx <grid.shape[0] and 0<= yyy <grid.shape[1]:
            grid[xxx,yyy] -= empty_odds
    sat = 127
    grid[grid> sat]= sat
    grid[grid<-sat]= -sat
    
    return grid




def texture_mapping(texture_map,now,rgb_img_name,depth_img_name,res):
    
    depth_img = imageio.imread(depth_img_name)
    rgb_img = imageio.imread(rgb_img_name)
    
    
    dd = -0.00304 * depth_img + 3.31
    depth = 1.03/dd
    
    # filter out invalid value
    idx = np.where(np.logical_and(depth<6.0,depth>0.001))
    
    Intr_matrix = np.zeros((3,3))
    Intr_matrix[0,0] = 585.05108211
    Intr_matrix[0,1] = 0
    Intr_matrix[0,2] = 242.94140713
    Intr_matrix[1,1] = 585.05108211
    Intr_matrix[1,2] = 315.83800193
    Intr_matrix[2,2] = 1
    
    
    Roc = np.array([[0,   -1,   0],
                    [0,    0,   -1],
                    [1,    0,   0]])
        
    # body to camera
    T_c_b = np.zeros((4,4))
    T_c_b[0,3] = 0.18
    T_c_b[1,3] = 0.005
    T_c_b[2,3] = 0.36
    T_c_b[3,3] = 1
    Roll = 0
    Pitch = 0.36
    Yaw = 0.021
    Rz = np.array([[np.cos(Yaw),-np.sin(Yaw),0],
                    [np.sin(Yaw), np.cos(Yaw),0],
                    [0          ,           0,1]])
    Ry = np.array([[np.cos(Pitch), 0, np.sin(Pitch)],
                    [0            , 1,             0],
                    [-np.sin(Pitch),0, np.cos(Pitch)]])
    Rx = np.array([[1,             0,             0],
                   [0,  np.cos(Roll), -np.sin(Roll)],
                   [0,  np.sin(Roll),  np.cos(Roll)]])
    T_c_b[0:3,0:3] = (Rz.dot(Ry)).dot(Rx)
        
    
    # body to world
    T_b_w = np.array([[np.cos(now[2]), -np.sin(now[2]), 0, now[0]],
                       [np.sin(now[2]),  np.cos(now[2]), 0, now[1]],
                       [0             ,               0, 1, now[2]],
                       [0             ,               0, 0,      1]])
    
    # camera to world = camera to body body to world
    #T_c_w = T_c_b.dot(T_b_w)
    T_c_w = T_b_w.dot(T_c_b)
    
    Rwc = T_c_w[0:3,0:3]
        
    Pwc = np.zeros((3,1))
    Pwc[0] = T_c_w[0,3]
    Pwc[1] = T_c_w[1,3]
    Pwc[2] = T_c_w[2,3]
        
            
    Extri_matrix = np.zeros((4,4))
    Extri_matrix[0:3,0:3] = Roc.dot(Rwc.T)
    Extri_matrix[0:3,3:4] = -(Roc.dot(Rwc.T)).dot(Pwc)
    Extri_matrix[3,3] = 1
    
    
    for i in range(len(idx[0])):

        rgb_i = (idx[0][i] * 526.37 + dd[idx[0][i],idx[1][i]] *(-4.5 *1750.46) + 19276.0)/585.051
        rgb_j = (idx[1][i] * 526.37 + 16662.0)/585.051
      
        pixels = np.array([rgb_i, rgb_j, 1]).T
        
        Zo = depth[idx[0][i],idx[1][i]]
        
        can_proj_matrix = np.zeros((3,3))
        can_proj_matrix[0,0] = 1/Zo
        can_proj_matrix[1,1] = 1/Zo
        can_proj_matrix[2,2] = 1/Zo
        
        optical_frame_tmp = (np.linalg.inv(can_proj_matrix).dot(np.linalg.inv(Intr_matrix))).dot(pixels)
        optical_frame = np.zeros((4,1))
        optical_frame[0:3] = optical_frame_tmp.reshape(-1,1)
        optical_frame[3] = 1
        
        
        world_frame = (np.linalg.inv(Extri_matrix)).dot(optical_frame)
        world_frame[3]=1
        
        if world_frame[2]>1:
            continue
            
        xx = world_frame[0]/res + texture_map.shape[0]//2
        yy = world_frame[1]/res + texture_map.shape[1]//2
        
        if i==1:
            print(now)
            print(Pwc)
            print(xx,yy)
            
        texture_map[int(xx),int(yy),:] = rgb_img[int(rgb_i),int(rgb_j),:]

    
    return texture_map
    
def stratified_resample(W):
    N = len(W)
    rnd_tmp = (np.random.rand(len(W)) + range(len(W))) / len(W)
    result, i, j = [], 0, 0
    cumul_w = np.cumsum(W)
    while i < len(W):
        if rnd_tmp[i] >= cumul_w[j]:
            j += 1
        else:
            result.append(j)
            i += 1
    return np.array(result)   
        
   
if __name__ == '__main__':
    texture_map = np.zeros((500,500,3))
    now = np.array([0,0,0])
    res = 0.1
    rgb_name = 'dataRGBD//dataRGBD//RGB20//rgb20_1.png'
    depth_name = 'dataRGBD\dataRGBD\Disparity20\disparity20_1.png'
    depth_img = imageio.imread(depth_name)
    rgb_img = imageio.imread(rgb_name)
    
    plt.imshow(depth_img)
   
    
    dep_in_RGB_fr = depth_img.copy()
    dd = -0.00304 * depth_img + 3.31
    depth = 1.03/dd
    
    
    
    idx = np.where(np.logical_and(depth<0.7,depth>0.1))
    
    Intr_matrix = np.zeros((3,3))
    Intr_matrix[0,0] = 585.05108211
    Intr_matrix[0,1] = 0
    Intr_matrix[0,2] = 242.94140713
    Intr_matrix[1,1] = 585.05108211
    Intr_matrix[1,2] = 315.83800193
    Intr_matrix[2,2] = 1
    
    for i in range(len(idx[0])):
        print(i)
        rgb_i = (idx[0][i] * 526.37 + dd[idx[0][i],idx[1][i]] *(-4.5 *1750.46) + 19276.0)/585.051
        rgb_j = (idx[1][i] * 526.37 + 16662.0)/585.051
        
        pixels = np.array([rgb_i, rgb_j, 1]).T
        
        Zo = depth[idx[0][i],idx[1][i]]
        
        can_proj_matrix = np.zeros((3,4))
        can_proj_matrix[0,0] = 1/Zo
        can_proj_matrix[1,1] = 1/Zo
        can_proj_matrix[2,2] = 1/Zo
        
        optical_frame = (np.linalg.pinv(can_proj_matrix).dot(np.linalg.inv(Intr_matrix))).dot(pixels)
        
        Roc = np.array([[0, -1.0,   0],
                        [0,    0,   -1.0],
                        [1,    0,   0]])
        
        # body to camera
        T_c_b = np.zeros((4,4))
        T_c_b[0,3] = 0.18
        T_c_b[1,3] = 0.005
        T_c_b[2,3] = 0.36
        T_c_b[3,3] = 1
        Roll = 0
        Pitch = 0.36
        Yaw = 0.021
        Rz = np.array([[np.cos(Yaw),-np.sin(Yaw),0],
                       [np.sin(Yaw), np.cos(Yaw),0],
                       [0          ,           0,1]])
        Ry = np.array([[np.cos(Pitch), 0, np.sin(Pitch)],
                       [0            , 1,             0],
                       [-np.sin(Pitch),0, np.cos(Pitch)]])
        Rx = np.array([[1,             0,             0],
                       [0,  np.cos(Roll), -np.sin(Roll)],
                       [0,  np.sin(Roll),  np.cos(Roll)]])
        T_c_b[0:3,0:3] = (Rz.dot(Ry)).dot(Rx)
        
    
        # body to world
        T_b_w = np.array([[np.cos(now[2]), -np.sin(now[2]), 0, now[0]],
                          [np.sin(now[2]),  np.cos(now[2]), 0, now[1]],
                          [0             ,               0, 1, now[2]],
                          [0             ,               0, 0,      1]])
    
        # camera to body = camera to body body to world
        T_c_w = np.linalg.inv(T_b_w).dot(T_c_b)
        
    
        Rwc = T_c_w[0:3,0:3]
        
        Pwc = np.zeros((3,1))
        Pwc[0] = T_c_w[0,3]
        Pwc[1] = T_c_w[1,3]
        Pwc[2] = T_c_w[2,3]
        
        Extri_matrix = np.zeros((4,4))
        Extri_matrix[0:3,0:3] = Roc.dot(Rwc.T)
        Extri_matrix[0:3,3:4] = -(Roc.dot(Rwc.T)).dot(Pwc)
        Extri_matrix[3,3] = 1
        
        world_frame = np.linalg.inv(Extri_matrix).dot(optical_frame)
        
        xx = world_frame[0]/res + texture_map.shape[0]//2
        yy = world_frame[1]/res + texture_map.shape[1]//2
        
        texture_map[int(xx),int(yy),:] = rgb_img[idx[0][i],idx[1][i],:]
        
  
    
    #exture_mapping(texture_map,now,rgb_name,depth_name)

  
