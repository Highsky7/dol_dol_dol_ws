cmake_minimum_required(VERSION 3.0.2)
project(wall_processing)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_conversions
)

## Find PCL (with GPU modules enabled)
find_package(PCL REQUIRED COMPONENTS
  common
  io
  gpu_filters
  gpu_segmentation
  gpu_containers
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Add PCL compile definitions (if any)
add_definitions(${PCL_DEFINITIONS})

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_conversions
  DEPENDS PCL
)

## wall_detect_node (GPU-accelerated)
add_executable(wall_detect_node
  src/wall_detect_node.cpp
)
target_link_libraries(wall_detect_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## wall_compress_node (GPU-accelerated)
add_executable(wall_compress_node
  src/wall_compress_node.cpp
)
target_link_libraries(wall_compress_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Install targets (optional)
install(TARGETS wall_detect_node wall_compress_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
