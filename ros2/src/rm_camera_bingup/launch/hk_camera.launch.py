from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('rm_camera_bringup')
    params_file = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    return LaunchDescription([
        Node(
            package='rm_camera_bringup',
            executable='hk_camera_node',
            name='hk_camera_node',
            output='screen',
            parameters=[params_file],
        )
    ]) 