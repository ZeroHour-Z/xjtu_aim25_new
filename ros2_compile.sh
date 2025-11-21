#!/bin/bash
cd ~/DX_autoaim/ros2
colcon build --merge-install --symlink-install --parallel-workers 4 --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON