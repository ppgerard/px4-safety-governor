#!/usr/bin/env python3

import math
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseArray, Pose
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition


class SafetyGovernorRepulsiveField(Node):
    """OFFBOARD velocity governor with simple exponential repulsive field (NED).

    Behavior:
    - Nominally flies back-and-forth between two waypoints at 8 m/s.
    - Adds a repulsive velocity away from the nearest person:
        v_rep_mag = k * exp(-alpha * d) for d < influence_radius, else 0.
      Direction is away from the closest person (XY plane).
    - If a person is closer than close_radius (default 2 m), clamp total speed to 3 m/s.
    - Always globally cap to 8 m/s.
    - Publishes people positions in body frame for visualization.

        Assumptions:
        - People positions are a static list in local NED defined in code (no topic subscription).
        - Repulsion uses horizontal (XY) distance, consistent with common safety zones.
    """

    def __init__(self) -> None:
        super().__init__('safety_governor_repulsive_field')

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

        # Default/static people positions in local NED [x, y, z] (z down)
        self.people_ned: List[np.ndarray] = [
            np.array([5.0, 1.0, 0.0], dtype=float),
            np.array([10.0, -1.5, 0.0], dtype=float),
            np.array([15.0, 2.5, 0.0], dtype=float),
        ]

        # Waypoints (NED)
        self.wp_a = np.array([-10, 0.0, -2.0], dtype=float)   # start
        self.wp_b = np.array([30.0, 0.0, -2.0], dtype=float)  # end
        self.current_target = self.wp_b.copy()
        self.wp_switch_threshold = 1.0  # m

        # Repulsive field settings
        self.k_rep = 5.0           # m/s (scale of repulsion)
        self.alpha_rep = 1.0       # 1/m (decay with distance)
        self.influence_radius = 5.0  # m (no influence at or beyond this)
        self.close_radius = 3.0      # m (apply 3 m/s clamp inside this)

        # Speed caps
        self.v_cap_free = 8.0  # m/s global
        self.v_cap_near = 3.0  # m/s when inside close_radius

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

        self.get_logger().info('SafetyGovernorRepulsiveField started (NED, OFFBOARD velocity).')

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

        # Swap endpoints if reached
        if np.linalg.norm(self.current_target - p) < self.wp_switch_threshold:
            if np.allclose(self.current_target, self.wp_b):
                self.current_target = self.wp_a.copy()
            else:
                self.current_target = self.wp_b.copy()

        nav_target = self.current_target.copy()

        # Nominal velocity toward target at global cap
        dir_vec = nav_target - p
        dir_norm = float(np.linalg.norm(dir_vec))
        if dir_norm > 1e-6:
            v_nom = (dir_vec / dir_norm) * self.v_cap_free
        else:
            v_nom = np.zeros(3, dtype=float)

        # Repulsion (XY) from people in front (along goal direction) inside influence radius
        v_rep = np.zeros(3, dtype=float)
        if self.people_ned:
            p_xy = p[:2]
            distances = [float(np.linalg.norm(p_xy - q[:2])) for q in self.people_ned]
            d_min = min(distances)
            if d_min <= self.close_radius:
                v_max = self.v_cap_near
            else:
                v_max = self.v_cap_free

            # Forward direction (XY) toward navigation target
            goal_xy = nav_target[:2] - p_xy
            goal_xy_norm = float(np.linalg.norm(goal_xy))
            goal_dir_xy = goal_xy / goal_xy_norm if goal_xy_norm > 1e-6 else None

            for q, d in zip(self.people_ned, distances):
                if 1e-6 < d < self.influence_radius:
                    # Vector from drone to person
                    to_person_xy = q[:2] - p_xy
                    # Apply only if person lies in front (positive dot with goal direction)
                    if goal_dir_xy is not None and np.dot(goal_dir_xy, to_person_xy) <= 0.0:
                        continue
                    mag = self.k_rep * math.exp(-self.alpha_rep * d) * v_max
                    v_rep[:2] += (p_xy - q[:2]) / d * mag
        else:
            v_max = self.v_cap_free

        # Combine nominal and repulsion
        v_cmd = v_nom + v_rep
        # Global cap and near-person cap
        v_cmd = self._limit_speed(v_cmd, v_max)

        # Publish setpoint and people rel body
        self.publish_velocity_setpoint(v_cmd)
        self.publish_people_relative_body(p)

    def _limit_speed(self, v: np.ndarray, v_max: float) -> np.ndarray:
        # Clamp velocity magnitude to the provided contextual cap (v_max).
        if not math.isfinite(v_max) or v_max <= 0.0:
            return v.copy()
        speed = float(np.linalg.norm(v))
        if speed > v_max and speed > 1e-6:
            return v * (v_max / speed)
        return v.copy()

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
    node = SafetyGovernorRepulsiveField()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
