cmake_minimum_required(VERSION 3.5)
project(rm_camera_bringup)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  include
  /opt/MVS/include
)

link_directories(
        /opt/MVS/lib/64
        /usr/local/lib
        )

add_executable(hk_camera_node
  src/hk_camera_node.cpp
  src/camera.cpp
)

add_executable(video_image_node
  src/video_image_node.cpp
)

target_link_libraries(hk_camera_node
  ${OpenCV_LIBS}
  glog
  pthread
  MvCameraControl
  libMvCameraControl.so
)

target_link_libraries(video_image_node
  ${OpenCV_LIBS}
  pthread
)

ament_target_dependencies(hk_camera_node rclcpp sensor_msgs cv_bridge std_msgs)
ament_target_dependencies(video_image_node rclcpp sensor_msgs cv_bridge std_msgs)

install(TARGETS hk_camera_node video_image_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package() 