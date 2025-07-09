cmake_minimum_required(VERSION 3.0.2)
project(grade_speed_control)

## C++17 이상이라면 -std=c++17 로 바꿔도 됩니다.
add_definitions(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
)

catkin_package()   # 라이브러리/헤더 공개 없음

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(linear_vel_publisher
  src/linear_vel_publisher.cpp
)

target_link_libraries(linear_vel_publisher
  ${catkin_LIBRARIES}
)

add_dependencies(linear_vel_publisher
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)