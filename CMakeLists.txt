cmake_minimum_required(VERSION 2.8)

PROJECT(Circle)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

#aux_source_directory(. DIR_SRCS)

#add_executable(circle ${DIR_SRCS})
add_executable(circle main.cpp)
target_link_libraries(circle ${OpenCV_LIBS})




