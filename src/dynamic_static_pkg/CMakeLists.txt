cmake_minimum_required(VERSION 3.0.2)
project(dynamic_static_pkg)

## C++11 사용
add_compile_options(-std=c++11)

## find catkin and all required components
find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation    # 메시지 정의로부터 헤더 생성
  message_filters
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  visualization_msgs
)

## 메시지 파일 선언
add_message_files(
  FILES
    TrackedObjects.msg
)

## 메시지 코드 생성
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

## catkin 패키지 선언
catkin_package(
  CATKIN_DEPENDS
    roscpp
    message_runtime       # 런타임에 생성 메시지 헤더 필요
    message_filters
    tf2_ros
    tf2_geometry_msgs
    geometry_msgs         # geometry_msgs를 런타임 의존으로 선언
    visualization_msgs
)

## 인클루드 디렉토리
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## 노드 빌드
add_executable(dynamic_static_classifier
  src/dynamic_static_classifier.cpp
)

## 메시지 의존 보장
add_dependencies(dynamic_static_classifier
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## 링크 라이브러리
target_link_libraries(dynamic_static_classifier
  ${catkin_LIBRARIES}
)
