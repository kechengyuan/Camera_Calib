CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(calib)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calib main.cpp)
target_link_libraries(calib ${OpenCV_LIBS})
