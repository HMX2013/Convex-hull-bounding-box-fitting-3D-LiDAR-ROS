cmake_minimum_required(VERSION 2.8.3)
project(convex_optimize_fitting)

find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  pcl_ros
  roscpp
  tf
)

find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS pcl_ros autoware_msgs
)

link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(SOURCE_FILES
  nodes/convex_optimize_fitting/convex_optimize_fitting.cpp
)

add_executable(convex_optimize_fitting ${SOURCE_FILES})

add_dependencies(convex_optimize_fitting
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(convex_optimize_fitting
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)