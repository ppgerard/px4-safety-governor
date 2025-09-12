#!/usr/bin/env python3

import math
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseArray, Pose
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleStatus,
)


class SafetyGovernor(Node):
    """
    Minimal safety governor for a PX4-controlled quadcopter in OFFBOARD velocity mode.

    - Follows a simple back-and-forth line trajectory using a P position controller.
    - Limits commanded velocity and acceleration based on distance to closest person (horizontal and vertical separately).
    - Publishes relative positions of static people in the body frame as PoseArray.

    Frame convention: PX4 local NED for positions and velocities.
    """

    def __init__(self) -> None:
        super().__init__('safety_governor')

        # --- QoS profiles (aligned with PX4 examples) ---
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

        # --- Static people positions in local NED (meters) ---
        # [x, y, z] with z down
        self.people_ned: List[np.ndarray] = [
            np.array([5.0, 1.0, 0.0], dtype=float),
            np.array([10.0, -1.5, 0.0], dtype=float),
            np.array([15.0, 2.5, 0.0], dtype=float),
        ]

        # --- Trajectory (back-and-forth) in NED ---
        self.wp_a = np.array([0.0, 0.0, -5.0], dtype=float)
        self.wp_b = np.array([20.0, 0.0, -5.0], dtype=float)
        self.current_target = self.wp_b.copy()
        self.wp_switch_threshold = 0.5  # m

        # --- Simple P gains for position -> velocity ---
        self.kp_xy = 0.5
        self.kp_z = 0.5

        # --- Avoidance when within this horizontal radius to a person (m) ---
        self.avoid_radius = 3.0  # hard no-entry radius (m)
        # Additional avoidance shaping
        self.guard_band = 1.0     # start skimming and add tangent within [avoid_radius, avoid_radius+guard_band]
        self.k_tangent = 0.8      # tangential velocity magnitude near the boundary (m/s)
        self.k_push = 1.5         # outward push gain when inside (m/s per meter of penetration)

        # --- Velocity limits (proportional to distance, with absolute caps) ---
        # v_max_xy = min(v_cap_xy, kv_xy * d_h_min); v_max_z = min(v_cap_z, kv_z * d_v_min)
        self.kv_xy = 0.6
        self.kv_z = 0.6
        self.v_cap_xy = 3.0  # m/s
        self.v_cap_z = 1.5   # m/s (down positive)

        # --- Acceleration limits as slew-rate on velocity (per-axis / grouped) ---
        # a_max_xy = min(a_cap_xy, ka_xy * d_h_min); a_max_z = min(a_cap_z, ka_z * d_v_min)
        self.ka_xy = 0.8
        self.ka_z = 0.8
        self.a_cap_xy = 2.0  # m/s^2 applied to horizontal speed change
        self.a_cap_z = 1.0   # m/s^2 applied to vertical speed change

        # --- State ---
        self.vehicle_position: VehicleLocalPosition | None = None
        self.vehicle_status: VehicleStatus | None = None
        self.prev_v_cmd = np.zeros(3, dtype=float)
        # Track nav/arming state like the example
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # --- Publishers ---
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', self.qos_pub)
        self.pub_traj_sp = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', self.qos_pub)
        self.pub_people_rel = self.create_publisher(PoseArray, 'people/relative_body', self.qos_pub)

        # --- Subscribers ---
        self.sub_pos = self.create_subscription(
            VehicleLocalPosition, 'fmu/out/vehicle_local_position_v1', self.cb_vehicle_pos, self.qos_sub
        )
        self.sub_status = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status_v1', self.cb_vehicle_status, self.qos_sub
        )

        # --- Control timer (20 Hz) ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.cmdloop_callback)

        self.get_logger().info('SafetyGovernor node started (NED, OFFBOARD velocity).')

    # --- Callbacks ---
    def cb_vehicle_pos(self, msg: VehicleLocalPosition) -> None:
        self.vehicle_position = msg

    def cb_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    # --- Core loop ---
    def cmdloop_callback(self) -> None:
        # Always publish OffboardControlMode regularly
        self.publish_offboard_mode()

        if self.vehicle_position is None:
            return

        # Current position in NED
        p = np.array([self.vehicle_position.x, self.vehicle_position.y, self.vehicle_position.z], dtype=float)

        # Switch waypoint when close
        if np.linalg.norm(self.current_target - p) < self.wp_switch_threshold:
            self.current_target = self.wp_a.copy() if np.allclose(self.current_target, self.wp_b) else self.wp_b.copy()

        # Baseline P controller: position error -> velocity command
        err = self.current_target - p
        v_cmd = np.array([
            self.kp_xy * err[0],
            self.kp_xy * err[1],
            self.kp_z * err[2],
        ], dtype=float)

        # Adapt trajectory to go around people with a hard 3 m no-entry policy
        v_cmd = self._adapt_velocity_around_people(p, v_cmd)

        # Safety limits based on distance to nearest person (horizontal and vertical separately)
        d_h_min = self._min_horizontal_distance(p)
        d_v_min = self._min_vertical_distance(p)

        v_max_xy = min(self.v_cap_xy, self.kv_xy * d_h_min)
        v_max_z = min(self.v_cap_z, self.kv_z * d_v_min)

        # Clamp velocities: horizontal magnitude and vertical component separately
        v_cmd = self._limit_velocity(v_cmd, v_max_xy, v_max_z)

        # Acceleration (slew-rate) limiting
        a_max_xy = min(self.a_cap_xy, self.ka_xy * d_h_min)
        a_max_z = min(self.a_cap_z, self.ka_z * d_v_min)
        v_cmd = self._limit_accel(self.prev_v_cmd, v_cmd, a_max_xy, a_max_z, self.dt)

        # Publish limited velocity to PX4 as TrajectorySetpoint (velocity only)
        # Publish continuously to maintain OFFBOARD stream reliability
        self.publish_velocity_setpoint(v_cmd)

        # Publish relative positions of people in body frame
        self.publish_people_relative_body(p)

        # Update state
        self.prev_v_cmd = v_cmd

    # --- Helpers ---
    def _min_horizontal_distance(self, p: np.ndarray) -> float:
        if not self.people_ned:
            return float('inf')
        dists = [math.hypot(p[0] - q[0], p[1] - q[1]) for q in self.people_ned]
        return min(dists)

    def _closest_person_xy(self, p: np.ndarray) -> tuple[np.ndarray, float] | None:
        if not self.people_ned:
            return None
        # Return vector r from person to p (in XY) and distance
        best = None
        best_d = float('inf')
        for q in self.people_ned:
            r = np.array([p[0] - q[0], p[1] - q[1]], dtype=float)
            d = float(np.hypot(r[0], r[1]))
            if d < best_d:
                best_d = d
                best = r
        if best is None:
            return None
        return best, best_d

    def _adapt_velocity_around_people(self, p: np.ndarray, v_cmd: np.ndarray) -> np.ndarray:
        res = v_cmd.copy()
        closest = self._closest_person_xy(p)
        if closest is None:
            return res
        r_xy, d = closest
        if not math.isfinite(d) or d < 1e-6:
            return res
        safety = self.avoid_radius
        guard = max(0.0, self.guard_band)

        # Unit outward normal from person to vehicle in XY
        n = r_xy / d
        # Choose tangent direction aligned with current desired XY velocity
        v_xy = np.array([res[0], res[1]], dtype=float)
        if np.allclose(v_xy, 0.0):
            # Small forward tangent bias to avoid deadlock
            v_xy = np.array([1e-3, 0.0])
        t_cw = np.array([n[1], -n[0]])
        t_ccw = -t_cw
        t = t_cw if float(np.dot(v_xy, t_cw)) >= float(np.dot(v_xy, t_ccw)) else t_ccw

        v_n = float(np.dot(v_xy, n))
        v_t_vec = v_xy - v_n * n

        if d <= safety:
            # Inside the safety circle: no inward motion; push outward and skim tangentially
            v_n_out = max(0.0, v_n)
            push = self.k_push * (safety - d)
            v_xy_new = v_n_out * n + self.k_tangent * t + push * n
        elif d < safety + guard:
            # In guard band: do not move inward, bias tangentially to go around
            v_n_out = max(0.0, v_n)
            v_xy_new = v_n_out * n + v_t_vec + self.k_tangent * t
        else:
            return res

        res[0] = v_xy_new[0]
        res[1] = v_xy_new[1]
        return res

    def _min_vertical_distance(self, p: np.ndarray) -> float:
        if not self.people_ned:
            return float('inf')
        dists = [abs(p[2] - q[2]) for q in self.people_ned]
        return min(dists)

    def _limit_velocity(self, v: np.ndarray, v_max_xy: float, v_max_z: float) -> np.ndarray:
        v_lim = v.copy()
        # Horizontal
        v_h = math.hypot(v[0], v[1])
        if math.isfinite(v_max_xy) and v_max_xy >= 0.0 and v_h > v_max_xy and v_h > 1e-4:
            scale = v_max_xy / v_h
            v_lim[0] *= scale
            v_lim[1] *= scale
        # Vertical
        if math.isfinite(v_max_z) and v_max_z >= 0.0:
            v_lim[2] = float(np.clip(v[2], -v_max_z, v_max_z))
        return v_lim

    def _limit_accel(self, v_prev: np.ndarray, v_target: np.ndarray, a_max_xy: float, a_max_z: float, dt: float) -> np.ndarray:
        dv = v_target - v_prev
        # Horizontal slew on magnitude change
        dv_h = np.array([dv[0], dv[1], 0.0])
        mag_dv_h = math.hypot(dv_h[0], dv_h[1])
        max_dv_h = max(0.0, a_max_xy) * dt if math.isfinite(a_max_xy) else float('inf')
        if mag_dv_h > max_dv_h and mag_dv_h > 1e-6:
            dv_h *= (max_dv_h / mag_dv_h)
        # Vertical slew (per-axis)
        max_dv_z = max(0.0, a_max_z) * dt if math.isfinite(a_max_z) else float('inf')
        dv_z = float(np.clip(dv[2], -max_dv_z, max_dv_z))
        return np.array([v_prev[0] + dv_h[0], v_prev[1] + dv_h[1], v_prev[2] + dv_z], dtype=float)

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
        # Mark unused fields as NaN to indicate velocity-only control
        nan = float('nan')
        sp.position[0] = nan
        sp.position[1] = nan
        sp.position[2] = nan
        # Only velocity is used
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
        # Leave other fields default/zero
        self.pub_traj_sp.publish(sp)

    def publish_people_relative_body(self, p_ned: np.ndarray) -> None:
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'base_link'

        # Heading (yaw) from local position if available (rad, NED), rotate world->body by -yaw
        yaw = 0.0
        try:
            yaw = float(self.vehicle_position.heading)
        except Exception:
            pass

        cy = math.cos(-yaw)
        sy = math.sin(-yaw)

        for person in self.people_ned:
            r_world = person - p_ned  # vector from vehicle to person in NED
            # Rotate into body frame (x-forward, y-right, z-down in NED convention)
            bx = cy * r_world[0] - sy * r_world[1]
            by = sy * r_world[0] + cy * r_world[1]
            bz = r_world[2]

            pose = Pose()
            pose.position.x = float(bx)
            pose.position.y = float(by)
            pose.position.z = float(bz)
            # Orientation not used; leave identity (0,0,0,1)
            pose.orientation.w = 1.0
            pa.poses.append(pose)

        self.pub_people_rel.publish(pa)

    # No arming/offboard commands here; follow example strategy


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
