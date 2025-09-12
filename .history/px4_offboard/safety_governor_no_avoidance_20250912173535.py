#!/usr/bin/env python3

import math
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseArray, Pose
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition


class SafetyGovernorNoAvoidance(Node):
    """OFFBOARD velocity governor without geometric avoidance (NED).

    - Follows a simple back-and-forth line.
    - Applies a simple speed limit near people (no acceleration limiting).
    - Publishes people positions in the body frame.
    - 8 m/s max, 3.0 m/s à moins de 4 m d'un humain (default; configurable via ROS params).
    """

    def __init__(self) -> None:
        super().__init__('safety_governor_no_avoidance')

        # QoS similar to PX4 examples
        self.qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Static people positions in local NED coordinates [x, y, z] (z is down).
        # If you modify these, also update default.sdf world and visualizer.py accordingly.
        self.people_ned: List[np.ndarray] = [
            np.array([5.0, 1.0, 0.0], dtype=float),
            np.array([10.0, -1.5, 0.0], dtype=float),
            np.array([15.0, 2.5, 0.0], dtype=float),
        ]

        # Base line waypoints in NED
        self.wp_a = np.array([0.0, 0.0, -5.0], dtype=float)   # start
        self.wp_b = np.array([20.0, 0.0, -5.0], dtype=float)  # end
        self.current_target = self.wp_b.copy()                 # active endpoint
        self.wp_switch_threshold = 1  # m; swap ends when close

    # Simple near-people speed limiting (no proportional, no accel limiting)
    # Declare parameters (one-liners to avoid duplication)
    self.safety_radius = float(self.declare_parameter('safety_radius', 4.0).value)   # m
    self.v_cap_near = float(self.declare_parameter('v_cap_near', 3.0).value)        # m/s
    self.v_cap_free = float(self.declare_parameter('v_cap_free', 8.0).value)        # m/s
    self.debug_safety = bool(self.declare_parameter('debug_safety', False).value)   # bool
        self._last_debug_log_ns = 0

        # State
        self.vehicle_position = None

        # Pub/Sub
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', self.qos_pub)
        self.pub_traj_sp = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', self.qos_pub)
        self.pub_people_rel = self.create_publisher(PoseArray, 'people/relative_body', self.qos_pub)
        self.sub_pos = self.create_subscription(
            VehicleLocalPosition, 'fmu/out/vehicle_local_position_v1', self.cb_vehicle_pos, self.qos_sub
        )

        # Control timer (20 Hz)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.cmdloop_callback)

        self.get_logger().info('SafetyGovernorNoAvoidance started (NED, OFFBOARD velocity).')

    # Callbacks
    def cb_vehicle_pos(self, msg: VehicleLocalPosition) -> None:
        self.vehicle_position = msg

    # Core loop
    def cmdloop_callback(self) -> None:
        # Maintain Offboard mode stream
        self.publish_offboard_mode()
        if self.vehicle_position is None:
            return

        # Current position
        p = np.array([self.vehicle_position.x, self.vehicle_position.y, self.vehicle_position.z], dtype=float)

        # Swap to the other endpoint if we reached current target
        if np.linalg.norm(self.current_target - p) < self.wp_switch_threshold:
            if np.allclose(self.current_target, self.wp_b):
                self.current_target = self.wp_a.copy()
            else:
                self.current_target = self.wp_b.copy()

        nav_target = self.current_target.copy()

        # Horizontal (XY) distance to nearest person (cylindrical gating)
        # Note: with the default path at z = -5 m and people at z = 0, 3D distance is >= 5 m
        # which prevents slowdown. Using XY distance ensures slowdown when horizontally close.
        if self.people_ned:
            p_xy = p[:2]
            d_min = min(float(np.linalg.norm(p_xy - q[:2])) for q in self.people_ned)
        else:
            d_min = float('inf')

        # Choose cap based on horizontal distance membership
        v_max = self.v_cap_near if d_min <= self.safety_radius else self.v_cap_free
        # Command velocity along the line/lookahead direction at magnitude v_max
        dir_vec = nav_target - p
        dir_norm = float(np.linalg.norm(dir_vec))
        if dir_norm > 1e-6 and v_max > 1e-6:
            v_cmd = (dir_vec / dir_norm) * v_max
        else:
            v_cmd = np.zeros(3, dtype=float)

        # Optional debug log once per second
        if self.debug_safety:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_debug_log_ns > 1_000_000_000:
                self._last_debug_log_ns = now_ns
                self.get_logger().info(
                    f"safety: d_min_xy={d_min:.2f}m v_max={v_max:.2f} m/s pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
                )

        # Publish velocity setpoint
        self.publish_velocity_setpoint(v_cmd)

        # Publish people relative positions in body frame
        self.publish_people_relative_body(p)

    def publish_offboard_mode(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.pub_offboard_mode.publish(msg)

    def publish_velocity_setpoint(self, v_ned: np.ndarray) -> None:
        sp = TrajectorySetpoint()
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        nan = float('nan')
        sp.position[0] = nan
        sp.position[1] = nan
        sp.position[2] = nan
        sp.velocity[0] = float(v_ned[0])
        sp.velocity[1] = float(v_ned[1])
        sp.velocity[2] = float(v_ned[2])
        sp.acceleration[0] = nan
        sp.acceleration[1] = nan
        sp.acceleration[2] = nan
        sp.jerk[0] = nan
        sp.jerk[1] = nan
        sp.jerk[2] = nan
        sp.yaw = nan
        sp.yawspeed = nan
        self.pub_traj_sp.publish(sp)

    def publish_people_relative_body(self, p_ned: np.ndarray) -> None:
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'base_link'

        # Heading (yaw) if available (rad, NED)
        yaw = 0.0
        try:
            yaw = float(self.vehicle_position.heading)
        except Exception:
            pass
        cy = math.cos(-yaw)
        sy = math.sin(-yaw)

        for person in self.people_ned:
            r_world = person - p_ned
            bx = cy * r_world[0] - sy * r_world[1]
            by = sy * r_world[0] + cy * r_world[1]
            bz = r_world[2]
            pose = Pose()
            pose.position.x = float(bx)
            pose.position.y = float(by)
            pose.position.z = float(bz)
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.pub_people_rel.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGovernorNoAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
