#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    """Launch the no-avoidance safety governor and optional visualizer."""

    with_visualizer = LaunchConfiguration('with_visualizer', default='true')

    return LaunchDescription([
        DeclareLaunchArgument('with_visualizer', default_value='true', description='Launch RViz visualizer node'),

        Node(
            package='px4_offboard',
            executable='safety_governor_no_avoidance',
            name='safety_governor_no_avoidance',
            output='screen',
        ),
        # Visualizer is optional; disable with with_visualizer:=false
        Node(
            package='px4_offboard',
            executable='visualizer',
            name='visualizer',
            output='screen',
            # Visualizer uses absolute topics; namespace unnecessary
            condition=IfCondition(with_visualizer),
        ),
    ])
