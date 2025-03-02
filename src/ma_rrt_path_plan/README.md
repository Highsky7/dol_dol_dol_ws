# RRT with multiple remote goals.

This repository contains the [ROS](https://www.ros.org/) package with code implementation of RRT-based path planning algorithm suitable for exploration of a trackdrive circuit according to the rules of Formula Student Driverless [competition](https://www.formulastudent.de/fsg/) in Germany. This algorithm is based on my contribution to [E-gnition](https://www.egnition.hamburg/), a FS Team of Technical University Hamburg.

![Basic animation of the approach](https://github.com/egnitionHamburg/ma_rrt_path_plan/blob/master/anim/drive.gif "RRT with multiple remote goals")

A brief introduction to the main steps of the proposed algorithm is given in my Master's thesis presentation [(direct timestamp)](https://youtu.be/eOevF5jFSoc?t=475).

[<img src="https://img.youtube.com/vi/eOevF5jFSoc/hqdefault.jpg" width="50%">](https://youtu.be/eOevF5jFSoc)

### Notes
- The algorithm does not utilize the cones' color information, but instead a simple logic is used, which rewards the branches with cones from both sides (see findBestBranch(...) method), and penalizes the branches having cones only from one side. With cone classification information a better rating system can be implemented and applied.
- Unfortunately I wasn't able to test and see this algorithm working on real hardware, a FS Driverless car, so I am excited if you can bring it the reality on any mobile robot and share some videos with me (see section **Usage**)

### Usage
- Exploration of [FSG'18 trackdrive circuit](https://www.youtube.com/watch?v=kjssdifs0DQ) in Gazebo.
- Exploration of [FSG'17 trackdrive circuit](https://www.youtube.com/watch?v=jJAjrCig3yE) in Gazebo.
- Exploration of [ISCC'21 in Korea Track Mission](https://youtu.be/tF_vQYmdT30) in Real HW (ERP-42) - Jeong Boin
- *Your video (feel free to pull-request a link with it here).*

## Inputs, outputs, params

#### Inputs
- rospy.Subscriber("/map", Map, ...)
- rospy.Subscriber("/odometry", Odometry, ...)

#### Outputs
- rospy.Publisher("/waypoints", WaypointsArray, ...)

#### Outputs (Visuals)
- rospy.Publisher("/visual/tree_marker_array", MarkerArray, ...)
- rospy.Publisher("/visual/best_tree_branch", Marker, ...)
- rospy.Publisher("/visual/filtered_tree_branch", Marker, ...)
- rospy.Publisher("/visual/delaunay_lines", Marker, ...)
- rospy.Publisher("/visual/waypoints", MarkerArray, ...)

#### Parameters to tune (main)
- planDistance = 12 m - Maximum length of tree branch
- expandDistance = 1 m - Length of tree node/step
- expandAngle = 20 deg - constraining angle for next tree nodes
- coneObstacleSize = 1.1 m - size of obstacles derived from cone position
- coneTargetsDistRatio = 0.5 - ratio to frontConesDist for deriving remote cone goals

## Licence

### MIT
- Use me and my code in any way you want
- Keep the names and the same licence


---
Edit. FROZEN Team from Sookmyung Women's University
- Jeong Boin : Decision - System Architecture Design, Motion Planning(+local purepursuit), Module Integration
- Lee Sunmyung (github.com/leesunmyung) : Perception - LiDAR Sensor Programming, Obstacle Recognition & Distance Data Callback
- Shin Sujin : Control - PID, PD control(longtitudinal Control), Purepursuit(latitudinal Control)


#### Inputs
- rospy.Subscriber("/track", Track, ...)
- rospy.Subscriber("/newwaypoints", WaypointsArray, ...)

#### Outputs
- rospy.Publisher("/waypoints", WaypointsArray, ...)

#### Outputs (Visuals)
- Same as the original

#### Parameters to tune (main)
- Same as the original

#### Architecture & Configuration + etc...
![슬라이드1](https://user-images.githubusercontent.com/54930076/161677596-2bfe92a0-5c2f-4ac8-85b1-d842ed102bf3.png)
![슬라이드4](https://user-images.githubusercontent.com/54930076/161677605-ccdb9f20-6522-40ed-ba9a-39ffc710892c.png)
![슬라이드5](https://user-images.githubusercontent.com/54930076/161677607-345cf3fc-0bc3-40f3-aed6-14f59db2832f.png)
![슬라이드9](https://user-images.githubusercontent.com/54930076/161677609-d44d2be7-3a31-4589-ab2a-41158e3c4518.png)
![슬라이드11](https://user-images.githubusercontent.com/54930076/161677613-8dade1d3-556b-4f90-8e0e-32c2bea1a5ff.png)
![슬라이드16](https://user-images.githubusercontent.com/54930076/161677616-059680ce-35cf-423c-96d4-4956f2b85691.png)
![슬라이드19](https://user-images.githubusercontent.com/54930076/161677620-2d4e4891-233a-4812-8f22-0f97c17c286d.png)

