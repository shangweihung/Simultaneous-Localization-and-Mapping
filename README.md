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
![image](https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/dataset_20.gif,{width=400px height=400px})  
![image](https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/dataset_21.gif,{width=400px height=400px})  
![image](https://github.com/shangweihung/Simultaneous-Localization-and-Mapping/blob/master/Demo_gif/testset.gif,{width=400px height=400px})  
