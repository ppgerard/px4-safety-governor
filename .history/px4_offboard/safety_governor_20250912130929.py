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

            # --- Safety and circumnavigation parameters ---
            self.safety_radius = 3.0     # no-entry around people (m)
            self.circle_radius = 3.25    # follow this radius when circumnavigating (m)
            self.circle_mode = False
            self.circle_center = np.zeros(2, dtype=float)
            self.circle_dir = 1          # +1 CCW, -1 CW
            self.circle_step = 1.0       # arc length lookahead along circle (m)
        # Detour planning params (pure geometry, no repulsive gains)
        self.detour_active = False
        self.detour_wp = np.zeros(3, dtype=float)
        self.detour_eps = 0.05  # small epsilon to step just outside the circle

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

        # Select navigation target: detour waypoint (if active) or the main waypoint
        nav_target = self.detour_wp if self.detour_active else self.current_target

        # If no detour is active, check if the straight path intersects any person's safety circle
        if not self.detour_active:
            planned = self._plan_detour_waypoint(p, nav_target)
            if planned is not None:
                self.detour_wp = planned
                nav_target = self.detour_wp
                self.detour_active = True

        # Complete detour when reached
        if self.detour_active:
            # use horizontal distance for detour completion
            if math.hypot(nav_target[0] - p[0], nav_target[1] - p[1]) < self.wp_switch_threshold:
                self.detour_active = False
                nav_target = self.current_target

        # Baseline P controller: position error -> velocity command
        err = nav_target - p
        v_cmd = np.array([
            self.kp_xy * err[0],
            self.kp_xy * err[1],
            self.kp_z * err[2],
        ], dtype=float)

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

    # --- Circle navigation helpers ---
    def _first_circle_on_path(self, p_xy: np.ndarray, t_xy: np.ndarray, radius: float) -> tuple[bool, np.ndarray | None]:
        """Return whether the segment p->t intersects any person's circle of given radius.
        If yes, return True and the center of the earliest intersection along the path.
        """
        if not self.people_ned:
            return False, None
        best_proj = float('inf')
        best_center = None
        d = t_xy - p_xy
        L2 = float(np.dot(d, d))
        for c in self.people_ned:
            c_xy = np.array([c[0], c[1]], dtype=float)
            # distance from center to segment
            dist = self._point_to_segment_distance(c_xy, p_xy, t_xy)
            if dist < radius - 1e-6:
                # projection for ordering along the path
                if L2 < 1e-12:
                    proj = 0.0
                else:
                    proj = float(np.dot(c_xy - p_xy, d) / np.sqrt(L2))
                if proj < best_proj:
                    best_proj = proj
                    best_center = c_xy
        return (best_center is not None), best_center

    @staticmethod
    def _point_to_segment_distance(c_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
        ab = b_xy - a_xy
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(c_xy - a_xy))
        t = float(np.dot(c_xy - a_xy, ab) / ab2)
        t = max(0.0, min(1.0, t))
        proj = a_xy + t * ab
        return float(np.linalg.norm(c_xy - proj))

    def _tangent_points_to_circle(self, P: np.ndarray, C: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the two tangent points from P to circle centered C radius R (2D)."""
        v = P - C
        d2 = float(np.dot(v, v))
        d = math.sqrt(max(d2, 0.0))
        if d <= R + 1e-6:
            # Inside or on circle: return two opposite points; fall back to radial point on circle
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

    def _circle_direction_for_entry(self, P: np.ndarray, T: np.ndarray, C: np.ndarray) -> int:
        """Choose +1 (CCW) or -1 (CW) so that motion from P->T aligns with circle tangent at T."""
        r = T - C
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            return 1
        t_ccw = np.array([-r[1], r[0]]) / r_norm
        t_cw = -t_ccw
        dir_vec = T - P
        d_ccw = float(np.dot(dir_vec, t_ccw))
        d_cw = float(np.dot(dir_vec, t_cw))
        return 1 if d_ccw >= d_cw else -1

    def _entry_score(self, P: np.ndarray, T: np.ndarray, C: np.ndarray, direction: int, target_xy: np.ndarray) -> float:
        """Heuristic score: smaller is better. Combines approach alignment and initial clearance toward target."""
        r = T - C
        r_norm = max(1e-6, float(np.linalg.norm(r)))
        t_dir = np.array([-r[1], r[0]]) / r_norm
        if direction < 0:
            t_dir = -t_dir
        approach = T - P
        approach_norm = max(1e-6, float(np.linalg.norm(approach)))
        align = 1.0 - float(np.dot(approach / approach_norm, t_dir))  # 0 when perfectly aligned
        # Favor the tangent that initially points roughly toward target
        to_target = target_xy - T
        to_target_norm = max(1e-6, float(np.linalg.norm(to_target)))
        head = 1.0 - float(np.dot(t_dir, to_target / to_target_norm))
        return align + 0.3 * head

    def _plan_detour_waypoint(self, p: np.ndarray, t: np.ndarray) -> np.ndarray | None:
        """
        If the straight path from p to t intersects any person's safety circle (XY),
        compute a single lateral detour waypoint just outside the circle and return it.
        Otherwise return None. Z of detour equals target Z.
        """
        if not self.people_ned:
            return None
        px, py = float(p[0]), float(p[1])
        tx, ty = float(t[0]), float(t[1])
        dx, dy = tx - px, ty - py
        L = math.hypot(dx, dy)
        if L < 1e-3:
            return None
        ux, uy = dx / L, dy / L  # unit along path
        # Perpendiculars
        left = (-uy, ux)
        right = (uy, -ux)

        best_wp = None
        best_proj = -1.0
        r = self.avoid_radius

        for c in self.people_ned:
            cx, cy = float(c[0]), float(c[1])
            # Projection of C onto the path segment
            wx, wy = cx - px, cy - py
            proj = wx * ux + wy * uy
            proj_clamped = min(max(proj, 0.0), L)
            qx, qy = px + ux * proj_clamped, py + uy * proj_clamped
            # Perpendicular distance to line
            dist = math.hypot(cx - qx, cy - qy)
            if dist >= r:
                continue  # no conflict with this person
            # Choose offset direction away from the person: sign by which side the center is on
            # cross_z = u x (C-Q) = ux*(cy-qy) - uy*(cx-qx)
            cross_z = ux * (cy - qy) - uy * (cx - qx)
            n = right if cross_z > 0 else left  # steer to the side away from the person
            # Amount to step laterally to clear the circle boundary
            lateral = (r - dist) + self.detour_eps
            wx_detour = qx + n[0] * lateral
            wy_detour = qy + n[1] * lateral
            # Prefer the person closest along the path (largest proj within [0,L])
            if proj_clamped > best_proj:
                best_proj = proj_clamped
                best_wp = np.array([wx_detour, wy_detour, t[2]], dtype=float)

        return best_wp

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
