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

add_executable(wall_detect_node src/wall_detect_node.cpp)
target_link_libraries(wall_detect_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(wall_compress_node src/wall_compress_node.cpp)
target_link_libraries(wall_compress_node ${catkin_LIBRARIES})

install(TARGETS wall_detect_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)