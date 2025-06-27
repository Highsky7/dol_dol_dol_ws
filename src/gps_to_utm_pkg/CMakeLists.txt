cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gps_to_utm_pkg LANGUAGES C CXX CUDA)

# GeographicLib CMake 모듈 경로 추가 (Debian/Ubuntu용)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/usr/share/cmake/geographiclib")

# Catkin과 GeographicLib 찾기
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)
find_package(GeographicLib REQUIRED)

# CMake 네이티브 CUDA 지원 설정
if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

catkin_package(
  CATKIN_DEPENDS
    roscpp
    roslib
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    tf2_geometry_msgs
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#-------------------------
# CPU 구현 노드들
#-------------------------

add_executable(global_yaw_estimator_cpp
  scripts/global_yaw_estimator.cpp
)
target_link_libraries(global_yaw_estimator_cpp
  ${catkin_LIBRARIES}
)

add_executable(gps_to_local_cartesian_cpp
  scripts/gps_to_local_cartesian.cpp
)
target_link_libraries(gps_to_local_cartesian_cpp
  ${catkin_LIBRARIES}
)

add_executable(gps_to_utm_node_cpp
  scripts/gps_to_utm_node.cpp
)
target_link_libraries(gps_to_utm_node_cpp
  ${catkin_LIBRARIES}
  ${GeographicLib_LIBRARIES}
)

add_executable(local_cartesian_path_publisher_cpp
  scripts/local_cartesian_path_publisher.cpp
)
target_link_libraries(local_cartesian_path_publisher_cpp
  ${catkin_LIBRARIES}
)

add_executable(ref_ant_vel_tf_broadcaster_cpp
  scripts/ref_ant_vel_tf_broadcaster.cpp
)
target_link_libraries(ref_ant_vel_tf_broadcaster_cpp
  ${catkin_LIBRARIES}
)

#-------------------------
# GPU 가속 ROI Path Publisher
#-------------------------

add_executable(roi_path_publisher_cpp
  scripts/roi_path_publisher.cu
)
set_target_properties(roi_path_publisher_cpp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(roi_path_publisher_cpp
  ${catkin_LIBRARIES}
)

#-------------------------
# 설치 설정
#-------------------------

install(TARGETS
  global_yaw_estimator_cpp
  gps_to_local_cartesian_cpp
  gps_to_utm_node_cpp
  local_cartesian_path_publisher_cpp
  ref_ant_vel_tf_broadcaster_cpp
  roi_path_publisher_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
