# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:52:58 2019

@author: Shang-Wei Hung
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys,cv2,os
from tool import show_lidar, mapping, texture_mapping, stratified_resample, first_scan
from map_utils import mapCorrelation
from scipy.special import expit



def load_encoder_data(dataset):
    with np.load("Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"]     # 4 x n encoder counts   #[FR; FL; RR; RL]
        encoder_stamps = data["time_stamps"] # encoder time stamps

    return encoder_counts, encoder_stamps

def load_imu_data(dataset):
    with np.load("Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    
    return imu_angular_velocity, imu_linear_acceleration, imu_stamps

def load_lidar_data(dataset):
    with np.load("Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    
    return lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamsp

def load_rgbd_data(dataset):
    with np.load("Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"]        # acquisition times of the rgb images
    
    RGB_path = f"dataRGBD\dataRGBD\RGB{dataset}"
    DEP_path = f"dataRGBD\dataRGBD\Disparity{dataset}"
    RGB_list = []
    for i in range(1,len(rgb_stamps)+1):
        RGB_list.append(RGB_path+f"\\rgb{dataset}_{i}.png")
    DEP_list = []
    for i in range(1,len(disp_stamps)+1):
        DEP_list.append(DEP_path+f"\\disparity{dataset}_{i}.png")
    
    return disp_stamps,rgb_stamps,RGB_list,DEP_list
    
    
    
def calculate_encoder(encoder_counts,encoder_stamps):
    '''
    Calculate Encoder + IMU  X,Y, Theta
    '''
    r_dis = ((encoder_counts[0,:]+encoder_counts[2,:])*0.0022/2).reshape(1,-1)
    l_dis = ((encoder_counts[1,:]+encoder_counts[3,:])*0.0022/2).reshape(1,-1)
    delta_t = [ encoder_stamps[i]-encoder_stamps[i-1] for i in range(1,len(encoder_stamps))]
    delta_t.append(delta_t[-1]) #fill to same length
    delta_t = np.asarray(delta_t) 
    v_left = l_dis/ delta_t
    v_right = r_dis/ delta_t
    v = (v_left+v_right)/2.0
    
    
    x=[0]
    y=[0]
    theta=[0]
    tmp=[]
    for i in range(1,len(encoder_stamps)):
        imu_idx = np.argmin(np.abs(encoder_stamps[i-1]-imu_stamps))
        tmp.append(imu_idx)
        omega= imu_angular_velocity[2,imu_idx]
   
        t_to = delta_t[i-1]
        x.append(x[i-1] + v[0][i-1] * t_to * math.sin(omega*t_to/2) / (omega*t_to/2) * math.cos(theta[i-1] + omega*t_to/2))
        y.append(y[i-1] + v[0][i-1] * t_to * math.sin(omega*t_to/2) / (omega*t_to/2) * math.sin(theta[i-1] + omega*t_to/2))
        theta.append(theta[i-1] + omega * t_to)
    
    #for i in range(0,len(x),50):
    #    plt.plot(x[i],y[i],'ro')
    #plt.show()
    
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    theta = np.array(theta).reshape(-1,1)
    encoder_pos = np.hstack((x,y,theta))

    return encoder_pos

def first_scan(lidar_ranges,lidar_stamsp,now,grid,lidar_angle):
    '''
    show first scan of lidar measurement
    '''
    lidar_scan_1st = lidar_ranges[:,0]
    lidar_ts_1st = lidar_stamsp[0]
    
    now = np.array([0,0,0])                                 # initial position
    grid = mapping(grid,lidar_scan_1st,now,res,lidar_angle) # first scan
    plt.imshow(grid)
    
    
    
def slam(X,W,encoder_stamps,encoder_pos,lidar_stamsp,lidar_ranges,rgb_stamps,disp_stamps,grid,res,Texture_map):
    
    #video = cv2.VideoWriter('dataset_20.avi', -1, 1, (grid.shape[0], grid.shape[1]), isColor=3)
    #video_text = cv2.VideoWriter('dataset_20_texture.avi', -1, 1, (grid.shape[0], grid.shape[1]), isColor=3)
    
    now_list = []
    pred_pos = encoder_pos[0]
    noise = np.array([0.02, 0.02, 0.5 * np.pi /180])
    for t, lidar in enumerate(lidar_stamsp[::50]):
        
        print('\n')
        print(t)
        scan = lidar_ranges[:,t*50]
        
        # Time alignment
        encoder_idx = np.argmin(np.abs(lidar-encoder_stamps))
        encoder_delta = encoder_pos[encoder_idx] - pred_pos 
        rgb_idx = np.argmin(np.abs(lidar-rgb_stamps)) 
        dep_idx = np.argmin(np.abs(lidar-disp_stamps))
        
        # predict
        noises = np.random.randn(N,3)*noise   
        X = X + encoder_delta + noises
        X[:,2] %= 2*math.pi
    
        
        x_im,y_im = np.arange(grid.shape[0]),np.arange(grid.shape[1])
        xs = ys = np.arange(-res*4,res*4+res,res)
        
        tem = np.zeros_like(grid)
        tem[grid>0] = 1
        tem[grid<0] = -1
        
        corr_max = []
        for i in range(len(X)):
            angles = lidar_angle + X[i,2]
            valid_scan = np.logical_and(scan>=0.1,scan<=30)
            ranges = scan[valid_scan]
            
            
            theta = angles[valid_scan] 
            # turn to Cartesian coordinate system
            xx, yy = ranges * np.cos(theta), ranges * np.sin(theta)
            # turn to cells
            xx = xx/res + grid.shape[0]//2
            yy = yy/res + grid.shape[1]//2
                
            cor = mapCorrelation(grid,x_im,y_im,np.vstack((xx,yy)), (X[i,0]+xs)/res, (X[i,1]+ys)/res)
            corr_max.append(np.max(cor))
            
            
        # update particle weights
        corr_max = W * np.array(corr_max)
        e_x = np.exp(corr_max-np.max(corr_max))
        W = e_x / e_x.sum()
        
        # find best particle
        best_particle = np.where(W == np.max(W))[0][0]
        now_best = X[best_particle].copy()
        now_best = now_best.ravel()
        test = now_best.copy()
        now_best[0]/= res
        now_best[1]/= res
         
 
        
        grid = mapping(grid,scan,now_best,res,lidar_angle)
        #Texture_map = texture_mapping(Texture_map, test, RGB_list[rgb_idx],DEP_list[dep_idx],res)    
        
        
        #img = np.zeros((grid_size, grid_size, 3))
        #img[:, :, 0] = grid 
        #img[:, :, 1] = grid 
        #img[:, :, 2] = grid 
        
        
        #cv2.circle(img,(int(now_best[1]) + grid.shape[1]//2,int(now_best[0]) + grid.shape[0]//2),4,(0,0,255),-1)
        #video.write(img.astype(np.uint8))
        
        #Texture_map_tmp = Texture_map.copy()
        #Texture_map_tmp = cv2.cvtColor(Texture_map_tmp.astype(np.uint8),cv2.COLOR_RGB2BGR)
        #cv2.circle(Texture_map_tmp,(int(now_best[1]) + grid.shape[1]//2,int(now_best[0]) + grid.shape[0]//2),4,(0,0,255),-1)
        #video_text.write(Texture_map_tmp)
 
        
        # resampling
        if sum(1/(W**2)) < (0.85 * N):
            idx = stratified_resample(W)
            W.fill(1.0 / N)
            X[:] = X[idx]
            
    
        now_list.append(now_best)
        pred_pos = encoder_pos[encoder_idx]
        

    
    #video.release()
    #video_text.release()
    
    return now_list,grid,Texture_map




if __name__ == '__main__':
    
    dataset = 21   #speficy the dataset you want to run
    #---------------------Load Data--------------------------
    encoder_counts, encoder_stamps = load_encoder_data(dataset)
    imu_angular_velocity,imu_linear_acceleration,imu_stamps = load_imu_data(dataset)
    lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamsp = load_lidar_data(dataset)
    disp_stamps,rgb_stamps,RGB_list,DEP_list = load_rgbd_data(dataset)
    
    
    encoder_pos = calculate_encoder(encoder_counts,encoder_stamps)
    
    # -------------------Initialization----------------------
    N = 100             # number of particles
    X = np.zeros((N,3)) # record pos of particles
    W = np.ones(N)/N    # partical weights
    res = 0.1           # resolution of the map
    grid_size = int(80/res)
    grid = np.zeros((grid_size,grid_size))
    
    Texture_map = np.zeros((grid_size,grid_size,3))
    lidar_angle = np.array([i* lidar_angle_increment.item(0) for i in range(-540,541)])
    
    #first_scan(lidar_ranges,lidar_stamsp,now,grid,lidar_angle)
    
    #-----SLAM(Particle filter+mapping+texture mapping)-----
    now_list,grid,Texture_map = slam(X,W,encoder_stamps,encoder_pos,lidar_stamsp,lidar_ranges,rgb_stamps,disp_stamps,grid,res,Texture_map)

    
    #------------------------Plot results---------------------
    result_tra = grid.copy()
    result_tra[grid>0] = 0.5
    result_tra[np.logical_and(0<=grid, grid<=0)] = 0.3
    result_tra[grid<0] = 0.1
    
    # slam results
    slam_result = result_tra.copy()
    now_list = np.array(now_list)
    plt.imshow(slam_result)
    x0,y0=np.round(now_list[:,0])+grid.shape[0]//2,np.round(now_list[:,1])+grid.shape[1]//2
    plt.plot(y0.astype(int),x0.astype(int),'r','-', 2.0)
    plt.show()
    
    # slam result + only encoders
    plt.imshow(slam_result)
    x0,y0=np.round(now_list[:,0])+grid.shape[0]//2,np.round(now_list[:,1])+grid.shape[1]//2
    plt.plot(y0.astype(int),x0.astype(int),'r','-', 2.0)
    x1,y1=np.round(encoder_pos[:,0]/res)+grid.shape[0]//2,np.round(encoder_pos[:,1]/res)+grid.shape[1]//2
    plt.plot(y1.astype(int),x1.astype(int),'b','-', 2.0)
    plt.show()
    
    
    # texture map result
    plt.figure()
    plt.imshow(Texture_map.astype(np.uint8))
    plt.show()
    





    