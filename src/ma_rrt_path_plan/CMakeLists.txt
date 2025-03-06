cmake_minimum_required(VERSION 2.8.3)
project(ma_rrt_path_plan)

# vehicle_msgs를 하위 디렉토리로 포함
add_subdirectory(vehicle_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  geometry_msgs
  message_filters
  message_generation
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS roscpp rospy std_msgs vehicle_msgs
#  DEPENDS system_lib
CATKIN_DEPENDS std_msgs sensor_msgs geometry_msgs message_filters message_runtime
)


include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/
  ${catkin_INCLUDE_DIRS}
)


install(DIRECTORY
    launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(PROGRAMS
    src/main.py
    src/MaRRTPathPlanNode.py
    src/ma_rrt.py
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
