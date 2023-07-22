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
set(RKNN_API_PATH /home/orangepi/source_code/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

#rga
set(RGA_PATH /home/orangepi/source_code/RK3588_3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)

# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread ")

add_executable(rknn_yolo 
		src/main.cpp
		src/base/rknn_infer_base.hpp
		src/base/rknn_infer_thread_base.hpp
		src/base/tools.hpp
		src/postprocess/rknn_postprocess.cpp
		src/postprocess/rknn_postprocess.h
		src/yolox/rknn_yolox.cpp
		src/yolox/rknn_yolox.h
		src/yolox/rknn_yolox_thread.cpp
		src/yolox/rknn_yolox_thread.h
		src/yolox/yolox_tools.hpp
)

target_link_libraries(rknn_yolo
	${RKNN_RT_LIB}
	${RGA_LIB}
	${OpenCV_LIBS}
	pthread
)
