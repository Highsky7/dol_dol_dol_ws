cmake_minimum_required(VERSION 3.0.2)
project(camera_lane_segmentation)

#############################################
## 1) 의존 패키지 검색                     ##
#############################################
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs       # 주로 헤더만 쓰이지만 python 노드에서도 필요
)

#############################################
## 2) catkin 패키지 설정                  ##
#############################################
catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    rospy
    sensor_msgs
    std_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
    tf2_geometry_msgs
)

#############################################
## 3) include 디렉터리 설정               ##
#############################################
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############################################
## 4) 파이썬 노드 설치 (스크립트)         ##
#############################################
## 필요할 때 여러 스크립트를 추가하세요.
catkin_install_python(PROGRAMS
  scripts/roboflow_final.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############################################
## 5) (컴파일 대상 없음 → 생략)           ##
#############################################
