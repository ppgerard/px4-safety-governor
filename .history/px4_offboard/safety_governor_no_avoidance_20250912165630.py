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

    - Follows a simple back-and-forth line using P position→velocity.
    - Applies a simple speed limit near people (no acceleration limiting).
    - Publishes people positions in the body frame.
    - 8 m/s max, 1.5 m/s à moins de 4 m d'un humain.
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
        self.wp_switch_threshold = 0.5  # m; swap ends when close
        self.base_lookahead = 1       # m along the line to smooth motion and set cruise speed

        # P controller gains (pos error -> vel command)
        self.kp_xy = 3
        self.kp_z = 0.8

    # Simple near-people speed limiting (no proportional, no accel limiting)
    self.safety_radius = 4.0   # m; inside this circle around any person, use the near cap
    self.v_cap_near = 1.5      # m/s when inside safety_radius
    self.v_cap_free = 8.0      # m/s otherwise (global cap)

        # State
        self.vehicle_position = None
        self.prev_v_cmd = np.zeros(3, dtype=float)

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

        # Base-line lookahead target for smooth, centered tracking
        p_xy = np.array([p[0], p[1]], dtype=float)
        a_xy = np.array([self.wp_a[0], self.wp_a[1]], dtype=float)
        b_xy = np.array([self.wp_b[0], self.wp_b[1]], dtype=float)
        d_line = b_xy - a_xy
        L = float(np.linalg.norm(d_line))
        if L > 1e-6:
            u_line = d_line / L
            s_on = float(np.dot(p_xy - a_xy, u_line))
            s_on = max(0.0, min(L, s_on))
            dir_sign = 1.0 if np.allclose(self.current_target, self.wp_b) else -1.0
            s_goal = max(0.0, min(L, s_on + dir_sign * self.base_lookahead))
            line_xy = a_xy + u_line * s_goal
            nav_target = np.array([line_xy[0], line_xy[1], self.current_target[2]], dtype=float)
        else:
            nav_target = self.current_target.copy()

        # P controller: position error -> velocity
        err = nav_target - p
        v_cmd = np.array([
            self.kp_xy * err[0],
            self.kp_xy * err[1],
            self.kp_z * err[2],
        ], dtype=float)

        # Horizontal distance to nearest person (circle in XY)
        if self.people_ned:
            d_h_min = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in self.people_ned)
        else:
            d_h_min = float('inf')

        # Choose cap based on circle membership; enforce on 3D norm
        v_max = self.v_cap_near if d_h_min <= self.safety_radius else self.v_cap_free
        v_cmd = self._limit_velocity(v_cmd, v_max)

        # Publish velocity setpoint
        self.publish_velocity_setpoint(v_cmd)

        # Publish people relative positions in body frame
        self.publish_people_relative_body(p)

        # Update state
        self.prev_v_cmd = v_cmd

    # Helpers
    def _limit_velocity(self, v: np.ndarray, v_max: float) -> np.ndarray:
        """Limit the 3D speed norm to v_max."""
        v_lim = v.copy()
        speed = float(np.linalg.norm(v_lim))
        if math.isfinite(v_max) and v_max >= 0.0 and speed > v_max and speed > 1e-4:
            v_lim *= (v_max / speed)
        return v_lim

    # Note: acceleration limiting removed per requirements

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
