cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gps_to_utm_pkg LANGUAGES C CXX CUDA)

# GeographicLib CMake 모듈 경로 (Debian/Ubuntu)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/usr/share/cmake/geographiclib")

# 1) Catkin 패키지
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
  ublox_msgs
)

# 2) 기타 의존성
find_package(GeographicLib REQUIRED)
find_package(CUDA REQUIRED)

# 3) CUDA 표준 설정 (CMake 내장)
if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

enable_language(CUDA)

# 4) catkin_package 선언
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
    ublox_msgs
)

# 5) 인클루드 디렉토리
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# -------------------------
# CPU 노드들
# -------------------------
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


add_executable(ublox_fix_processor_cpp
  scripts/ublox_fix_processor.cpp)

target_link_libraries(ublox_fix_processor_cpp
  ${catkin_LIBRARIES})


add_executable(purepursuit_cpp
  scripts/purepursuit.cpp
)
target_link_libraries(purepursuit_cpp
  ${catkin_LIBRARIES}
)


# add_executable(navpvt_global_yaw_estimator_cpp
#   scripts/navpvt_global_yaw_estimator.cpp
# )
# target_link_libraries(navpvt_global_yaw_estimator_cpp
#   ${catkin_LIBRARIES}
# )
# -------------------------
# GPU 가속 ROI Path Publisher
# -------------------------
cuda_add_executable(roi_path_publisher_cpp
  scripts/roi_path_publisher.cu
)

set_target_properties(roi_path_publisher_cpp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(roi_path_publisher_cpp
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
)


# -------------------------
# 의존성 설정
add_dependencies(global_yaw_estimator_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(gps_to_local_cartesian_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(gps_to_utm_node_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(local_cartesian_path_publisher_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ref_ant_vel_tf_broadcaster_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ublox_fix_processor_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(roi_path_publisher_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(purepursuit_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}
)

# -------------------------
# 설치 설정
# -------------------------
install(TARGETS
  global_yaw_estimator_cpp
  gps_to_local_cartesian_cpp
  gps_to_utm_node_cpp
  local_cartesian_path_publisher_cpp
  ref_ant_vel_tf_broadcaster_cpp
  roi_path_publisher_cpp
  ublox_fix_processor_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
