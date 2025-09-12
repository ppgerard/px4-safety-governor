#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

from re import M
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = position[0]
    pose_msg.pose.position.y = position[1]
    pose_msg.pose.position.z = position[2]
    return pose_msg


class PX4Visualizer(Node):
    def __init__(self):
        super().__init__("visualizer")

        # Configure subscritpions
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to both legacy and _v1 topic names to be robust across PX4 versions
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile,
        )
        self.attitude_sub_v1 = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude_v1",
            self.vehicle_attitude_callback,
            qos_profile,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile,
        )
        self.local_position_sub_v1 = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.vehicle_local_position_callback,
            qos_profile,
        )
        self.setpoint_sub = self.create_subscription(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            self.trajectory_setpoint_callback,
            qos_profile,
        )

        self.vehicle_pose_pub = self.create_publisher(
            PoseStamped, "/px4_visualizer/vehicle_pose", 10
        )
        self.vehicle_vel_pub = self.create_publisher(
            Marker, "/px4_visualizer/vehicle_velocity", 10
        )
        self.vehicle_path_pub = self.create_publisher(
            Path, "/px4_visualizer/vehicle_path", 10
        )
        self.setpoint_path_pub = self.create_publisher(
            Path, "/px4_visualizer/setpoint_path", 10
        )
        # People (cylinder) markers
        self.people_marker_pub = self.create_publisher(
            Marker, "/px4_visualizer/people_markers", 10
        )
        # Safety sphere marker (min distance visualization)
        self.safety_marker_pub = self.create_publisher(
            Marker, "/px4_visualizer/min_distance_sphere", 10
        )
        # Base trajectory (straight line) marker publisher
        self.base_traj_pub = self.create_publisher(
            Marker, "/px4_visualizer/base_trajectory", 10
        )

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_path_msg = Path()
        self.setpoint_path_msg = Path()

        # trail size
        self.trail_size = 1000

        # time stamp for the last local position update received on ROS2 topic
        self.last_local_pos_update = 0.0
        # time after which existing path is cleared upon receiving new
        # local position ROS2 message
        self.declare_parameter("path_clearing_timeout", -1.0)

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # Static people positions in ENU (map) frame, matching Gazebo world
        # Each entry: (x, y, z_center)
        self.people_radius = 0.35
        self.people_height = 7.5
        ground_center_z = self.people_height / 2.0
        self.people_positions_enu = [
            (1.0, 5.0, ground_center_z),
            (-1.5, 10.0, ground_center_z),
            (2.5, 15.0, ground_center_z),
        ]
        # safety sphere radius (meters)
        self.safety_radius = 1.5

        # Base trajectory endpoints in ENU (map) frame (straight A-B line)
        # PX4 NED waypoints: A(0,0,-5), B(20,0,-5) -> ENU: (y,x,-z) = (0,0,5) and (0,20,5)
        self.base_traj_A = np.array([0.0, 0.0, 5.0])
        self.base_traj_B = np.array([0.0, 20.0, 5.0])

    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg):
        path_clearing_timeout = (
            self.get_parameter("path_clearing_timeout")
            .get_parameter_value()
            .double_value
        )
        if path_clearing_timeout >= 0 and (
            (Clock().now().nanoseconds / 1e9 - self.last_local_pos_update)
            > path_clearing_timeout
        ):
            self.vehicle_path_msg.poses.clear()
        self.last_local_pos_update = Clock().now().nanoseconds / 1e9
        # NED (PX4) -> ENU (RViz) transformation
        # ENU.x = NED.y, ENU.y = NED.x, ENU.z = -NED.z
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def trajectory_setpoint_callback(self, msg):
        # Some controllers publish velocity-only setpoints with NaN positions.
        # Only update the setpoint path when positions are finite.
        if (
            len(msg.position) >= 3
            and np.isfinite(msg.position[0])
            and np.isfinite(msg.position[1])
            and np.isfinite(msg.position[2])
        ):
            # Convert NED -> ENU for visualization
            self.setpoint_position[0] = msg.position[1]
            self.setpoint_position[1] = msg.position[0]
            self.setpoint_position[2] = -msg.position[2]

    def create_arrow_marker(self, id, tail, vector):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "arrow"
        msg.id = id
        msg.type = Marker.ARROW
        msg.scale.x = 0.1
        msg.scale.y = 0.2
        msg.scale.z = 0.0
        msg.color.r = 0.5
        msg.color.g = 0.5
        msg.color.b = 0.0
        msg.color.a = 1.0
        dt = 0.3
        tail_point = Point()
        tail_point.x = tail[0]
        tail_point.y = tail[1]
        tail_point.z = tail[2]
        head_point = Point()
        head_point.x = tail[0] + dt * vector[0]
        head_point.y = tail[1] + dt * vector[1]
        head_point.z = tail[2] + dt * vector[2]
        msg.points = [tail_point, head_point]
        return msg

    def create_cylinder_marker(self, id, position, radius, height, color=(0.8, 0.2, 0.2)):
        # position: (x, y, z_center) in ENU (map) frame
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "people"
        msg.id = id
        msg.type = Marker.CYLINDER
        msg.scale.x = 2.0 * radius
        msg.scale.y = 2.0 * radius
        msg.scale.z = height
        msg.color.r = float(color[0])
        msg.color.g = float(color[1])
        msg.color.b = float(color[2])
        msg.color.a = 0.8
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        # Orientation identity (cylinder axis aligned with Z)
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        return msg

    def create_sphere_marker(self, id, center, radius, color=(0.9, 0.1, 0.1), alpha=0.15):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "safety"
        msg.id = id
        msg.type = Marker.SPHERE
        msg.scale.x = 2.0 * radius
        msg.scale.y = 2.0 * radius
        msg.scale.z = 2.0 * radius
        msg.color.r = float(color[0])
        msg.color.g = float(color[1])
        msg.color.b = float(color[2])
        msg.color.a = float(alpha)
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.w = 1.0
        return msg

    def append_vehicle_path(self, msg):
        self.vehicle_path_msg.poses.append(msg)
        if len(self.vehicle_path_msg.poses) > self.trail_size:
            del self.vehicle_path_msg.poses[0]

    def append_setpoint_path(self, msg):
        self.setpoint_path_msg.poses.append(msg)
        if len(self.setpoint_path_msg.poses) > self.trail_size:
            del self.setpoint_path_msg.poses[0]

    def cmdloop_callback(self):
        vehicle_pose_msg = vector2PoseMsg(
            "map", self.vehicle_local_position, self.vehicle_attitude
        )
        vehicle_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.vehicle_pose_pub.publish(vehicle_pose_msg)

        # Publish time history of the vehicle path
        self.vehicle_path_msg.header = vehicle_pose_msg.header
        self.append_vehicle_path(vehicle_pose_msg)
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # Publish time history of the vehicle path
        setpoint_pose_msg = vector2PoseMsg(
            "map", self.setpoint_position, self.vehicle_attitude
        )
        setpoint_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_path_msg.header = setpoint_pose_msg.header
        # Only publish/append setpoint path when it's finite (skip NaN velocity-only cases)
        if np.all(np.isfinite(self.setpoint_position)):
            self.append_setpoint_path(setpoint_pose_msg)
            self.setpoint_path_pub.publish(self.setpoint_path_msg)

        # Publish arrow markers for velocity
        velocity_msg = self.create_arrow_marker(
            1, self.vehicle_local_position, self.vehicle_local_velocity
        )
        self.vehicle_vel_pub.publish(velocity_msg)

        # Publish people cylinders as markers (static positions)
        colors = [
            (0.9, 0.2, 0.2),  # red
            (0.2, 0.6, 0.9),  # blue-ish
            (0.2, 0.8, 0.3),  # green
        ]
        for idx, pos in enumerate(self.people_positions_enu, start=1):
            cyl = self.create_cylinder_marker(
                id=idx,
                position=pos,
                radius=self.people_radius,
                height=self.people_height,
                color=colors[(idx - 1) % len(colors)],
            )
            self.people_marker_pub.publish(cyl)

        # Publish safety sphere centered at vehicle position
        safety_marker = self.create_sphere_marker(
            id=1,
            center=self.vehicle_local_position,
            radius=self.safety_radius,
            color=(0.9, 0.1, 0.1),
            alpha=0.15,
        )
        self.safety_marker_pub.publish(safety_marker)

        # Publish base straight trajectory (bold line) A->B
        base_line = Marker()
        base_line.action = Marker.ADD
        base_line.header.frame_id = "map"
        base_line.header.stamp = self.get_clock().now().to_msg()
        base_line.ns = "base_trajectory"
        base_line.id = 1
        base_line.type = Marker.LINE_STRIP
        base_line.scale.x = 0.2  # line width
        base_line.color.r = 0.0
        base_line.color.g = 0.0
        base_line.color.b = 0.0
        base_line.color.a = 1.0
        p1 = Point(); p1.x, p1.y, p1.z = self.base_traj_A
        p2 = Point(); p2.x, p2.y, p2.z = self.base_traj_B
        base_line.points = [p1, p2]
        self.base_traj_pub.publish(base_line)


def main(args=None):
    rclpy.init(args=args)

    px4_visualizer = PX4Visualizer()

    rclpy.spin(px4_visualizer)

    px4_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
