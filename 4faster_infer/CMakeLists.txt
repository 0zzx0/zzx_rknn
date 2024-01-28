cmake_minimum_required(VERSION 3.4.1)

project(faster_rknn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 是否开启ROS
option(USE_ROS "Enable ROS compilation" OFF) 

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

# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread ")

set(BASEFILES 	src/base/rknn_infer_base.hpp
				src/base/rknn_infer_thread_base.hpp
				src/base/tools.hpp
				src/postprocess/rknn_postprocess.cpp
				src/postprocess/rknn_postprocess.h
				src/preprocess/rknn_preprocess.cpp
				src/preprocess/rknn_preprocess.h
				
)
set(YOLOX 	src/apps/yolox/rknn_yolox.cpp
			src/apps/yolox/rknn_yolox.h
			src/apps/yolox/rknn_yolox_thread.cpp
			src/apps/yolox/rknn_yolox_thread.h
			src/apps/yolox/yolox_postprocess.cpp
			src/apps/yolox/yolox_postprocess.h
)
set(YOLOV8 	src/apps/yolov8-seg/rknn_yolov8_seg.cpp
			src/apps/yolov8-seg/rknn_yolov8_seg.h
			src/apps/yolov8-seg/yolov8_seg_postprocess.h
			src/apps/yolov8-seg/yolov8_seg_postprocess.cpp

)
add_executable(faster_rknn src/main.cpp ${BASEFILES} ${YOLOX} ${YOLOV8})

target_link_libraries(faster_rknn
	${RKNN_RT_LIB}
	${RGA_LIB}
	${OpenCV_LIBS}
	pthread
)

# 根据USE_ROS变量的值来决定是否添加ROS相关的编译目标  
if(USE_ROS) 
	find_package(catkin REQUIRED COMPONENTS  
	cv_bridge
	image_transport
	roscpp
	rospy
	sensor_msgs
	std_msgs
	message_generation  
	)   
	catkin_package(  
		CATKIN_DEPENDS cv_bridge image_transport roscpp rospy std_msgs sensor_msgs message_generation
	)
	add_message_files(
		FILES
		detection_result.msg
	)  
	
	# 添加ROS相关的源文件和目录  
	include_directories(include ${catkin_INCLUDE_DIRS})  
	link_directories(${catkin_LIBRARIES})  
	add_executable(faster_rknn_ros src/main_ros.cpp ${BASEFILES})  
	target_link_libraries(faster_rknn_ros 	${RKNN_RT_LIB}
											${RGA_LIB}
											${OpenCV_LIBS}
											${catkin_LIBRARIES}
											pthread
	)   
endif()


