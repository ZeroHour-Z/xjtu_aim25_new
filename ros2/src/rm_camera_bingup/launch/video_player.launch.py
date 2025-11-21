from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    topic_name = LaunchConfiguration('topic_name', default='/image_raw')
    video_path = LaunchConfiguration('video_path', default='/home/ovalene/DX_autoaim/video/blue/v1.avi')
    publish_rate = LaunchConfiguration('publish_rate', default='30.0')
    loop = LaunchConfiguration('loop', default='true')
    use_image_shm = LaunchConfiguration('use_image_shm', default='true')
    image_shm_name = LaunchConfiguration('image_shm_name', default='/image_raw_shm')
    image_shm_size = LaunchConfiguration('image_shm_size', default='8388672')

    return LaunchDescription([
        DeclareLaunchArgument('topic_name', default_value=topic_name),
        DeclareLaunchArgument('video_path', default_value=video_path),
        DeclareLaunchArgument('publish_rate', default_value=publish_rate),
        DeclareLaunchArgument('loop', default_value=loop),
        DeclareLaunchArgument('use_image_shm', default_value=use_image_shm),
        DeclareLaunchArgument('image_shm_name', default_value=image_shm_name),
        DeclareLaunchArgument('image_shm_size', default_value=image_shm_size),
        Node(
            package='rm_camera_bringup',
            executable='video_image_node',
            name='video_image_node',
            output='screen',
            parameters=[{
                'topic_name': topic_name,
                'video_path': video_path,
                'publish_rate': publish_rate,
                'loop': loop,
                'use_image_shm': use_image_shm,
                'image_shm_name': image_shm_name,
                'image_shm_size': image_shm_size,
            }]
        )
    ]) 