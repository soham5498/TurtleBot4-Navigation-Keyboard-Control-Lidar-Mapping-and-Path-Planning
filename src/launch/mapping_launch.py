
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom']
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/soham/ros2_ws/src/rviz/occupancy_grid.rviz']
        ),
        Node(
            package='mapping_package',
            executable='mapping_node',
            name='mapping_node',
            output='screen',
        ),
        Node(
            package='mapping_package',
            executable='path_planning_node',
            name='path_planning_node',
            output='screen'
        )
    ])


