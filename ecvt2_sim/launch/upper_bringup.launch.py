# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    upper_node = Node(
        package='ecvt2_sim',
        executable='upper_node',
        name='sim_ecvt_upper_node',
        output='screen'
    )

    return LaunchDescription([
        upper_node,
    ])
