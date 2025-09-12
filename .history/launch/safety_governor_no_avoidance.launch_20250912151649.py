#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    """Launch the no-avoidance safety governor and optional visualizer."""

    namespace = LaunchConfiguration('namespace', default='px4_offboard')
    with_visualizer = LaunchConfiguration('with_visualizer', default='true')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='px4_offboard', description='ROS namespace'),
        DeclareLaunchArgument('with_visualizer', default_value='true', description='Launch RViz visualizer node'),

        Node(
            package='px4_offboard',
            namespace=namespace,
            executable='safety_governor_no_avoidance',
            name='safety_governor_no_avoidance',
            output='screen',
        ),
        # Visualizer is optional; disable with with_visualizer:=false
        Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer',
            output='screen',
            parameters=[{'namespace': namespace}],
            condition=IfCondition(with_visualizer),
        ),
    ])
