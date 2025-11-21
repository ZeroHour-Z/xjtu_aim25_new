cmake_minimum_required(VERSION 3.8)
project(rm_dx_vision)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DDEBUGMODE)
add_compile_options(-Wall -Wextra -Wpedantic)

# ROS 2 build system
find_package(ament_cmake REQUIRED)
# 新增：ROS 2 依赖
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)

# Third-party dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(Threads REQUIRED)

# Try OpenVINO via CMake package first
set(HAVE_OPENVINO_TARGET FALSE)
find_package(OpenVINO QUIET COMPONENTS Runtime)
if(OpenVINO_FOUND)
	set(HAVE_OPENVINO_TARGET TRUE)
else()
	# Fallback to manual discovery
	find_path(OPENVINO_INCLUDE_DIR NAMES openvino/openvino.hpp)
	find_library(OPENVINO_RUNTIME_LIB NAMES openvino openvino_runtime libopenvino)
	if(OPENVINO_INCLUDE_DIR AND OPENVINO_RUNTIME_LIB)
		set(HAVE_OPENVINO_TARGET TRUE)
	endif()
endif()

# Local paths to sources and headers (inside this ROS2 package)

set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

set(ARMOR_SRC_DIR "${PROJECT_ROOT_DIR}/armor/src")
set(ARMOR_INCLUDE_DIR "${PROJECT_ROOT_DIR}/armor/include")

set(WINDMILL_SRC_DIR "${PROJECT_ROOT_DIR}/windmill/src")
set(WINDMILL_INCLUDE_DIR "${PROJECT_ROOT_DIR}/windmill/include")
set(PARAMS_INCLUDE_DIR "${PROJECT_ROOT_DIR}/params/include")
# 已移除: CAMERA_INCLUDE_DIR

# Sources (conditionally include OpenVINO source)
set(ARMOR_SOURCES
	${ARMOR_SRC_DIR}/AimAuto.cpp
	${ARMOR_SRC_DIR}/KalmanFilter.cpp
	${ARMOR_SRC_DIR}/tracker.cpp
	# Windmill energy modules
	${WINDMILL_SRC_DIR}/WMIdentify.cpp
	${WINDMILL_SRC_DIR}/WMPredict.cpp
	${WINDMILL_SRC_DIR}/angular_velocity_fitter.cpp
	${WINDMILL_SRC_DIR}/traditional_detection.cpp
	${WINDMILL_SRC_DIR}/WMInference.cpp
)
if(HAVE_OPENVINO_TARGET)
	list(APPEND ARMOR_SOURCES ${ARMOR_SRC_DIR}/OpenvinoInfer.cpp)
endif()

add_library(${PROJECT_NAME} STATIC ${ARMOR_SOURCES})

# Propagate availability macro
if(HAVE_OPENVINO_TARGET)
	target_compile_definitions(${PROJECT_NAME} PUBLIC OPENVINO_AVAILABLE=1)
else()
	target_compile_definitions(${PROJECT_NAME} PUBLIC OPENVINO_AVAILABLE=0)
endif()

# Add our own include directory first so that stub headers are picked up
include_directories(BEFORE ${ARMOR_INCLUDE_DIR})

# 公开头文件供依赖目标使用（包含外部params与windmill头）
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${ARMOR_INCLUDE_DIR}>
		$<BUILD_INTERFACE:${PARAMS_INCLUDE_DIR}>
		$<BUILD_INTERFACE:${WINDMILL_INCLUDE_DIR}>
		$<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Link dependencies
# OpenCV modern interface exports OpenCV::opencv or individual modules; use variable as fallback
if(TARGET OpenCV::opencv)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenCV::opencv)
else()
	target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()

# OpenVINO
if(OpenVINO_FOUND)
	target_link_libraries(${PROJECT_NAME} PUBLIC openvino::runtime)
elseif(HAVE_OPENVINO_TARGET)
	target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${OPENVINO_INCLUDE_DIR}>)
	target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENVINO_RUNTIME_LIB})
endif()

# Others

target_link_libraries(${PROJECT_NAME}
	PUBLIC
		Eigen3::Eigen
		Ceres::ceres
		glog::glog
		Threads::Threads
)

# 新增：可执行节点，订阅图像topic + String RX/TX
add_executable(armor_node ${CMAKE_CURRENT_SOURCE_DIR}/src/ros2_main.cpp)
ament_target_dependencies(armor_node rclcpp sensor_msgs cv_bridge std_msgs)
# 链接本包静态库
target_link_libraries(armor_node ${PROJECT_NAME})

# Install library
install(TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

# Install executable
install(TARGETS armor_node
	RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install public headers of this package
install(DIRECTORY ${ARMOR_INCLUDE_DIR}/
	DESTINATION share/${PROJECT_NAME}/include
)

# 安装launch与config（如果存在）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
install(DIRECTORY launch/
	DESTINATION share/${PROJECT_NAME}/launch
)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
install(DIRECTORY config/
	DESTINATION share/${PROJECT_NAME}/config
)
endif()

ament_package() 