cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolox)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置系统架构
set(LIB_ARCH aarch64)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# rknn api
set(RKNN_API_PATH /home/orangepi/rknn_1_6_0/runtime)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

#rga
set(RGA_PATH /home/orangepi/rknn_1_6_0/runtime/3rdparty/rga)
set(RGA_LIB ${RGA_PATH}/libs/Linux/gcc-aarch64/librga.so)
include_directories( ${RGA_PATH}/include)

add_executable(rknn_yolo
        src/main.cpp
		src/tools.hpp
		src/postprocess.hpp
)

target_link_libraries(rknn_yolo
	${RKNN_RT_LIB}
	${RGA_LIB}
	${OpenCV_LIBS}
)
