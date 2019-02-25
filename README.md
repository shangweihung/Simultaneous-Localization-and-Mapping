# Simultaneous-Localization-and-Mapping
Implement Simultaneous Localization and Mapping (SLAM) using odometry, inertial, 2-D laser range, and RGBD measurements from a differential-drive robot.  

# Introduction
Simultaneous Localization and Mapping (SLAM) is an extremely important algorithm in the field of robotics. It is a chicken-or-egg problem: a map is needed for localization and a pose estimate is needed for mapping. This algorithm can help robots or machines to understand the environment geometrically. This capability serves as a complementary function to the fancy deep learning applications. In this paper, I have implemented localization prediction and updating, occupancy grid mapping and texture mapping using encoders, IMU, lidar scan measurements and Kinect RGBD images.  

# Dependencies
* Python 3   
* cv2  
* numpy  
* matplotlib  
* math  

# Demo
<img src="https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/dataset_20.gif" height="280">  <img src="https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/dataset_21.gif" height="280">  <img src="https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/dataset_23_test.gif" height="280">  
**Red dot**: the current location of the robots.

# Functions:
1. main.py:    
  * calculate_encoder: calculate the discrete time model (x,y,theta) using encoder, IMU
  * slam: implement particle filter (predict and update)
   
2. tool.py:
  * stratified_resample: if the number of effective particles is less than a threshold, then perform stratified resampling.  
  * mapping: use log-odds to update the map.  
  * texture_mapping perform frame transformation to project the color pixel onto the floor. 
  
3. map_utils.py:	(**not written by myself**)
  * mapCorrelation: compute the 9x9 grid value around each particle to get map correlation and update the weights
  * bresenham2D: Bresenham's ray tracing algorithm in 2D
   
