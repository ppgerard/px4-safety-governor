#!/usr/bin/env python3

import math
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseArray, Pose
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition


class SafetyGovernor(Node):
    """Simple OFFBOARD velocity governor with circle-based avoidance (NED)."""

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
        self.wp_switch_threshold = 0.2  # m

        # --- Simple P gains for position -> velocity ---
        self.kp_xy = 0.5
        self.kp_z = 0.5

        # --- Minimal avoidance settings ---
        self.safety_radius = 1.5   # stay outside this radius
        self.circle_radius = 1.75  # follow this radius if avoidance is needed
        self.circle_active = False
        self.circle_center = np.zeros(2, dtype=float)
        self.circle_dir = 1  # +1 CCW, -1 CW
    self.circle_lookahead = 0.5  # meters along arc
    self.base_lookahead = 1.0    # meters along base line

        # --- Velocity limits (proportional to distance, with absolute caps) ---
        # v_max_xy = min(v_cap_xy, kv_xy * d_h_min); v_max_z = min(v_cap_z, kv_z * d_v_min)
        self.kv_xy = 0.6
        self.kv_z = 0.6
        self.v_cap_xy = 3.0  # m/s
        self.v_cap_z = 1.5   # m/s (down positive)

    # --- Acceleration limits as slew-rate on velocity ---
        self.ka_xy = 0.8
        self.ka_z = 0.8
        self.a_cap_xy = 2.0  # m/s^2 applied to horizontal speed change
        self.a_cap_z = 1.0   # m/s^2 applied to vertical speed change

    # --- State ---
    self.vehicle_position: VehicleLocalPosition | None = None
        self.prev_v_cmd = np.zeros(3, dtype=float)

        # --- Publishers ---
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', self.qos_pub)
        self.pub_traj_sp = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', self.qos_pub)
        self.pub_people_rel = self.create_publisher(PoseArray, 'people/relative_body', self.qos_pub)

        # --- Subscribers ---
        self.sub_pos = self.create_subscription(
            VehicleLocalPosition, 'fmu/out/vehicle_local_position_v1', self.cb_vehicle_pos, self.qos_sub
        )

        # --- Control timer (20 Hz) ---
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.cmdloop_callback)

        self.get_logger().info('SafetyGovernor node started (NED, OFFBOARD velocity).')

    # --- Callbacks ---
    def cb_vehicle_pos(self, msg: VehicleLocalPosition) -> None:
        self.vehicle_position = msg

    # no status callback needed

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

        # Decide nav target: default is base-line lookahead to naturally re-center
        p_xy = np.array([p[0], p[1]], dtype=float)
        t_xy = np.array([self.current_target[0], self.current_target[1]], dtype=float)
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

        if not self.circle_active:
            # Find first person whose safety circle intersects the straight path
            hit_center = None
            best_s = float('inf')
            seg = t_xy - p_xy
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1e-6:
                u = seg / seg_len
                for c in self.people_ned:
                    c_xy = np.array([c[0], c[1]], dtype=float)
                    # skip if behind start or beyond target
                    w = c_xy - p_xy
                    s = float(np.dot(w, u))
                    if s < 0.0 or s > seg_len:
                        continue
                    # check distance from segment to center
                    closest = p_xy + u * s
                    if float(np.linalg.norm(c_xy - closest)) < self.safety_radius:
                        if s < best_s:
                            best_s = s
                            hit_center = c_xy
            if hit_center is not None:
                # Compute tangent points from current position to the avoidance circle
                t1, t2 = self._tangent_points_to_circle(p_xy, hit_center, self.circle_radius)
                # choose tangent that better heads toward the target
                to_t = (t_xy - p_xy)
                if np.linalg.norm(to_t) < 1e-6:
                    pick = t1
                else:
                    v1 = t1 - p_xy
                    v2 = t2 - p_xy
                    pick = t1 if float(np.dot(v1, to_t)) >= float(np.dot(v2, to_t)) else t2
                nav_target = np.array([pick[0], pick[1], self.current_target[2]], dtype=float)
                # set circle state and direction consistent with tangent
                r = pick - hit_center
                t_ccw = np.array([-r[1], r[0]], dtype=float)
                v = pick - p_xy
                self.circle_dir = 1 if float(np.dot(v, t_ccw)) >= 0.0 else -1
                self.circle_center = hit_center
                self.circle_active = True
        else:
            # follow a point slightly ahead along the circle for smooth motion
            r = p_xy - self.circle_center
            R = max(1e-3, self.circle_radius)
            if float(np.linalg.norm(r)) < 1e-6:
                r = np.array([R, 0.0])
            ang = math.atan2(r[1], r[0])
            d_ang = self.circle_lookahead / R * float(self.circle_dir)
            ang_ahead = ang + d_ang
            ahead = self.circle_center + R * np.array([math.cos(ang_ahead), math.sin(ang_ahead)])
            nav_target = np.array([ahead[0], ahead[1], self.current_target[2]], dtype=float)
            # exit circle when the direct path ahead is clear (ignore first small step to avoid early exit)
            seg = t_xy - p_xy
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1e-6:
                u = seg / seg_len
                s_clear = max(0.5, float(self.circle_lookahead))  # meters ahead to ignore
                s_clear = min(s_clear, seg_len * 0.9)  # don't overshoot target
                p_ahead = p_xy + u * s_clear
            else:
                p_ahead = p_xy
            if not self._segment_intersects_circle(p_ahead, t_xy, self.circle_center, self.safety_radius):
                self.circle_active = False

        # Baseline P controller: position error -> velocity command
        err = nav_target - p
        v_cmd = np.array([
            self.kp_xy * err[0],
            self.kp_xy * err[1],
            self.kp_z * err[2],
        ], dtype=float)

        # Safety limits based on distance to nearest person (horizontal/vertical)
        if self.people_ned:
            d_h_min = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in self.people_ned)
            d_v_min = min(abs(p[2] - q[2]) for q in self.people_ned)
        else:
            d_h_min = d_v_min = float('inf')

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
    @staticmethod
    def _segment_intersects_circle(a_xy: np.ndarray, b_xy: np.ndarray, c_xy: np.ndarray, r: float) -> bool:
        ab = b_xy - a_xy
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(c_xy - a_xy)) < r
        t = float(np.dot(c_xy - a_xy, ab) / ab2)
        t = max(0.0, min(1.0, t))
        proj = a_xy + t * ab
        return float(np.linalg.norm(c_xy - proj)) < r

    @staticmethod
    def _tangent_points_to_circle(P: np.ndarray, C: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
        v = P - C
        d2 = float(np.dot(v, v))
        d = math.sqrt(max(d2, 0.0))
        if d <= R + 1e-6:
            # fallback: return the radial point on circle twice
            if d < 1e-6:
                r_hat = np.array([1.0, 0.0])
            else:
                r_hat = v / d
            T = C + R * r_hat
            return T, T
        l = (R * R) / d2
        m = R * math.sqrt(max(d2 - R * R, 0.0)) / d2
        perp = np.array([-v[1], v[0]])
        T1 = C + l * v + m * perp
        T2 = C + l * v - m * perp
        return T1, T2

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
