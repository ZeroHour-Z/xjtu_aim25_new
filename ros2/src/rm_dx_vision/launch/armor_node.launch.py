from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
	image_topic = LaunchConfiguration('image_topic', default='/image_raw')
	target_color = LaunchConfiguration('target_color', default='0')
	dt = LaunchConfiguration('dt', default='0.01')
	rx_topic = LaunchConfiguration('rx_topic', default='/autoaim/rx')
	tx_topic = LaunchConfiguration('tx_topic', default='/autoaim/tx')
	use_comm_shm = LaunchConfiguration('use_comm_shm', default='false')
	comm_shm_name = LaunchConfiguration('comm_shm_name', default='/rm_comm_shm')
	use_image_shm = LaunchConfiguration('use_image_shm', default='false')
	image_shm_name = LaunchConfiguration('image_shm_name', default='/image_raw_shm')
	image_shm_size = LaunchConfiguration('image_shm_size', default='8388672')
	vision_only = LaunchConfiguration('vision_only', default='true')

	return LaunchDescription([
		DeclareLaunchArgument('image_topic', default_value=image_topic),
		DeclareLaunchArgument('target_color', default_value=target_color),
		DeclareLaunchArgument('dt', default_value=dt),
		DeclareLaunchArgument('rx_topic', default_value=rx_topic),
		DeclareLaunchArgument('tx_topic', default_value=tx_topic),
		DeclareLaunchArgument('use_comm_shm', default_value=use_comm_shm),
		DeclareLaunchArgument('comm_shm_name', default_value=comm_shm_name),
		DeclareLaunchArgument('use_image_shm', default_value=use_image_shm),
		DeclareLaunchArgument('image_shm_name', default_value=image_shm_name),
		DeclareLaunchArgument('image_shm_size', default_value=image_shm_size),
		DeclareLaunchArgument('vision_only', default_value=vision_only),
		Node(
			package='rm_dx_vision',
			executable='armor_node',
			name='armor_node',
			output='screen',
			parameters=[{
				'image_topic': image_topic,
				'target_color': target_color,
				'dt': dt,
				'rx_topic': rx_topic,
				'tx_topic': tx_topic,
				'use_comm_shm': use_comm_shm,
				'comm_shm_name': comm_shm_name,
				'use_image_shm': use_image_shm,
				'image_shm_name': image_shm_name,
				'image_shm_size': image_shm_size,
				'vision_only': vision_only,
			}]
		)
	]) 