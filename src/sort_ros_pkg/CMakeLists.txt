cmake_minimum_required(VERSION 3.0.2)
project(sort_ros_pkg)

##-----------------------------------------------------------
## 1) catkin 패키지 & 메시지 패키지 의존 선언
##-----------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  visualization_msgs
  cv_bridge
  dynamic_static_pkg        # ← 우리가 만든 메시지 패키지
)

##-----------------------------------------------------------
## 2) 시스템 라이브러리
##-----------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

##-----------------------------------------------------------
## 3) catkin 패키지 메타정보
##    CATKIN_DEPENDS에 dynamic_static_pkg를 반드시 포함해야
##    CMake가 그 include 경로를 ${catkin_INCLUDE_DIRS}에 추가합니다.
##-----------------------------------------------------------
catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    visualization_msgs
    cv_bridge
    dynamic_static_pkg
)

##-----------------------------------------------------------
## 4) 인클루드 디렉토리 설정
##    ${catkin_INCLUDE_DIRS}에 dynamic_static_pkg의
##    devel/include 경로가 포함됩니다.
##-----------------------------------------------------------
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/devel/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

##-----------------------------------------------------------
## 5) 실행 파일 정의 & 링크
##-----------------------------------------------------------
add_executable(sort_ros_node
  src/main.cpp
  src/Sort.cpp
  src/SortRect.cpp
  src/SortRos.cpp
  src/Tracker.cpp
  src/TrackerState.cpp
  src/Hungarian.cpp
  src/PredictTrajectory.cpp
)

# 메시지/서비스가 먼저 생성되도록 의존성 설정
add_dependencies(sort_ros_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(sort_ros_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
