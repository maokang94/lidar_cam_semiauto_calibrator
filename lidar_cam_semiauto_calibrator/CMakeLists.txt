cmake_minimum_required(VERSION 3.10)
project(lidar_cam_semiauto_calibrator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree search sample_consensus segmentation filters)
find_package(Ceres REQUIRED)

add_definitions(${PCL_DEFINITIONS})

add_executable(lidar_cam_semiauto_calibrator
  src/lidar_camera_calib.cpp
  include/calib_app_helpers.hpp
  include/calib_utils.hpp
  include/common.h
  include/lidar_camera_calib.hpp
)

target_include_directories(lidar_cam_semiauto_calibrator
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(lidar_cam_semiauto_calibrator
  PRIVATE
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
)
