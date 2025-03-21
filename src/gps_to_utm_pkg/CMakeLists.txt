cmake_minimum_required(VERSION 2.8.3)
project(gps_to_utm_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gps_to_utm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
