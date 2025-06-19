cmake_minimum_required(VERSION 3.0.2)
project(tunnel_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_ros pcl_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(wall_filter_node src/wall_filter_node.cpp)
target_link_libraries(wall_filter_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS wall_filter_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
