cmake_minimum_required(VERSION 3.0.2)
project(yolo_ros_ros1)

## find_package에 yolo_msgs를 추가하여 메시지 패키지를 찾도록 합니다.
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  yolo_msgs
)

## catkin_package의 CATKIN_DEPENDS에도 yolo_msgs를 추가합니다.
## 이를 통해 다른 패키지가 이 패키지를 사용할 때 yolo_msgs 의존성을 알 수 있습니다.
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    cv_bridge
    yolo_msgs
)

## include_directories 설정은 C++ 코드가 있을 때 필요하므로 유지하거나 삭제해도 무방합니다.
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 실행할 파이썬 스크립트를 설치 대상으로 지정합니다.
catkin_install_python(PROGRAMS
  scripts/yolo_track_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)