#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='px4_offboard',
            executable='safety_governor_repulsive_field',
            name='safety_governor_repulsive_field',
            output='screen',
        )
    ])
