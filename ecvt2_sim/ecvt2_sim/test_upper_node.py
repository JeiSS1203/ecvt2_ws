#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
from typing import Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3Stamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory


# 상부체 토크 제어 대상(UPJ1,2,3,4,TOOLJ1) 속도 측정치
v_meas = np.zeros(5, dtype=float)


class UnifiedUpperNode(Node):
    def __init__(self, model, data, ctrl_array, joint_names, torque_joint_names, sensor_lookup):
        super().__init__('sim_ecvt2_upper_node')

        self.model = model
        self.data = data
        self.ctrl_array = ctrl_array
        self.joint_names = joint_names
        self.torque_joint_names = torque_joint_names
        self.sensor_lookup = sensor_lookup

        # -------- parameters --------
        self.declare_parameter('trajectory_topic', '/vp_sto_global_planner_node/actuated_reference')
        self.declare_parameter('traj_tracking_enabled', True)
        self.declare_parameter('traj_hold_last_point', True)
        self.declare_parameter('traj_use_position_feedback', True)
        self.declare_parameter('traj_position_kp', 1.5)
        self.declare_parameter('traj_velocity_ff_scale', 1.0)
        self.declare_parameter('traj_acceleration_ff_scale', 0.0)
        self.declare_parameter('traj_override_velocity_cmd', True)

        self.trajectory_topic = self.get_parameter('trajectory_topic').get_parameter_value().string_value
        self.traj_tracking_enabled = self.get_parameter('traj_tracking_enabled').get_parameter_value().bool_value
        self.traj_hold_last_point = self.get_parameter('traj_hold_last_point').get_parameter_value().bool_value
        self.traj_use_position_feedback = self.get_parameter('traj_use_position_feedback').get_parameter_value().bool_value
        self.traj_position_kp = self.get_parameter('traj_position_kp').get_parameter_value().double_value
        self.traj_velocity_ff_scale = self.get_parameter('traj_velocity_ff_scale').get_parameter_value().double_value
        self.traj_acceleration_ff_scale = self.get_parameter('traj_acceleration_ff_scale').get_parameter_value().double_value
        self.traj_override_velocity_cmd = self.get_parameter('traj_override_velocity_cmd').get_parameter_value().bool_value

        # -------- state for external velocity commands --------
        self.create_subscription(JointState, '/hanyang/velocity_cmd', self.velocity_callback, 10)

        # -------- state for planner trajectory tracking --------
        self.create_subscription(JointTrajectory, self.trajectory_topic, self.trajectory_callback, 10)
        self.active_traj: Optional[JointTrajectory] = None
        self.active_traj_start_time = None
        self.traj_is_active = False

        # optional debug
        self.last_q_ref = None
        self.last_qd_ref = None
        self.last_qdd_ref = None
        self.last_v_des = None

        # -------- publishers --------
        self.pub_joint_std = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_joint_hanyang = self.create_publisher(JointState, '/hanyang/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/hanyang/imu', 10)
        self.pub_pos = self.create_publisher(Vector3Stamped, '/hanyang/base_pos', 10)
        self.pub_com = self.create_publisher(Vector3Stamped, '/hanyang/com_pos', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.01, self.publish_sensor_data)

    # ------------------------------------------------------------------
    # External velocity command callback
    # ------------------------------------------------------------------
    def velocity_callback(self, msg: JointState):
        n_ctrl = len(self.ctrl_array)
        if len(msg.name) == len(msg.velocity) and len(msg.name) > 0:
            name_to_vel = dict(zip(msg.name, msg.velocity))
            matched = False
            for i, joint_name in enumerate(self.torque_joint_names):
                if joint_name in name_to_vel:
                    self.ctrl_array[i] = name_to_vel[joint_name]
                    matched = True
            if matched:
                return

        if len(msg.velocity) < n_ctrl:
            self.get_logger().warn(f"Expected at least {n_ctrl} velocities, got {len(msg.velocity)}")
            return
        self.ctrl_array[:n_ctrl] = msg.velocity[:n_ctrl]

    # ------------------------------------------------------------------
    # Planner trajectory callback
    # ------------------------------------------------------------------
    def trajectory_callback(self, msg: JointTrajectory):
        if not self.traj_tracking_enabled:
            return

        if len(msg.points) == 0:
            self.get_logger().warn("Received empty JointTrajectory")
            return

        expected = list(self.torque_joint_names)
        got = list(msg.joint_names)

        if got != expected:
            self.get_logger().warn(
                f"Trajectory joint names mismatch. expected={expected}, got={got}"
            )
            return

        self.active_traj = msg
        self.active_traj_start_time = self.get_clock().now()
        self.traj_is_active = True

        t_last = self._point_time_sec(msg.points[-1])
        self.get_logger().info(
            f"Received actuated reference trajectory: {len(msg.points)} points, duration={t_last:.3f}s"
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_sensor(self, name: str):
        if name not in self.sensor_lookup:
            self.get_logger().warn(f"Sensor '{name}' not found")
            return [0.0]
        addr, dim = self.sensor_lookup[name]
        return self.data.sensordata[addr:addr + dim]

    def _compute_world_com(self):
        masses = np.asarray(self.model.body_mass, dtype=float)
        xipos = np.asarray(self.data.xipos, dtype=float)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return np.zeros(3, dtype=float)
        return np.sum(masses[:, None] * xipos, axis=0) / total_mass

    @staticmethod
    def _point_time_sec(pt: JointTrajectoryPoint) -> float:
        return float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9

    def _read_torque_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q_meas = []
        qd_meas = []
        for name in self.torque_joint_names:
            pos_val = self._get_sensor(f"{name}_pos")
            vel_val = self._get_sensor(f"{name}_vel")
            q_meas.append(float(pos_val[0]) if len(pos_val) > 0 else 0.0)
            qd_meas.append(float(vel_val[0]) if len(vel_val) > 0 else 0.0)
        return np.array(q_meas, dtype=float), np.array(qd_meas, dtype=float)

    @staticmethod
    def _safe_array(data, dim: int, fill=0.0):
        if data is None or len(data) != dim:
            return np.full(dim, fill, dtype=float)
        return np.array(data, dtype=float)

    # ------------------------------------------------------------------
    # Cubic Hermite interpolation between two trajectory points
    # This is closer to a smooth spline playback than plain linear interpolation.
    # ------------------------------------------------------------------
    def _interpolate_segment(
        self,
        p0: JointTrajectoryPoint,
        p1: JointTrajectoryPoint,
        t_query: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = self._point_time_sec(p0)
        t1 = self._point_time_sec(p1)
        dt = max(1e-9, t1 - t0)
        tau = np.clip((t_query - t0) / dt, 0.0, 1.0)

        n = len(self.torque_joint_names)

        q0 = self._safe_array(p0.positions, n, fill=0.0)
        q1 = self._safe_array(p1.positions, n, fill=0.0)

        # If velocities are not provided, approximate from neighboring points.
        if len(p0.velocities) == n and len(p1.velocities) == n:
            v0 = np.array(p0.velocities, dtype=float)
            v1 = np.array(p1.velocities, dtype=float)
        else:
            slope = (q1 - q0) / dt
            v0 = slope.copy()
            v1 = slope.copy()

        # Cubic Hermite basis
        h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
        h10 = tau**3 - 2.0 * tau**2 + tau
        h01 = -2.0 * tau**3 + 3.0 * tau**2
        h11 = tau**3 - tau**2

        q = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1

        # derivative wrt tau, then /dt
        dh00 = 6.0 * tau**2 - 6.0 * tau
        dh10 = 3.0 * tau**2 - 4.0 * tau + 1.0
        dh01 = -6.0 * tau**2 + 6.0 * tau
        dh11 = 3.0 * tau**2 - 2.0 * tau

        qd = (dh00 * q0 + dh10 * dt * v0 + dh01 * q1 + dh11 * dt * v1) / dt

        # second derivative
        d2h00 = 12.0 * tau - 6.0
        d2h10 = 6.0 * tau - 4.0
        d2h01 = -12.0 * tau + 6.0
        d2h11 = 6.0 * tau - 2.0

        qdd = (d2h00 * q0 + d2h10 * dt * v0 + d2h01 * q1 + d2h11 * dt * v1) / (dt * dt)

        return q, qd, qdd

    # ------------------------------------------------------------------
    # Sample current active trajectory at elapsed time
    # ------------------------------------------------------------------
    def _sample_active_trajectory(self, elapsed_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.active_traj is None or len(self.active_traj.points) == 0:
            return None

        pts = self.active_traj.points
        n = len(self.torque_joint_names)

        t_last = self._point_time_sec(pts[-1])

        if elapsed_sec <= 0.0:
            p = pts[0]
            q = self._safe_array(p.positions, n, fill=0.0)
            qd = self._safe_array(p.velocities, n, fill=0.0)
            qdd = self._safe_array(p.accelerations, n, fill=0.0)
            return q, qd, qdd

        if elapsed_sec >= t_last:
            if self.traj_hold_last_point:
                p = pts[-1]
                q = self._safe_array(p.positions, n, fill=0.0)
                qd = np.zeros(n, dtype=float)
                qdd = np.zeros(n, dtype=float)
                return q, qd, qdd
            else:
                self.traj_is_active = False
                return None

        for i in range(len(pts) - 1):
            t0 = self._point_time_sec(pts[i])
            t1 = self._point_time_sec(pts[i + 1])
            if t0 <= elapsed_sec <= t1:
                return self._interpolate_segment(pts[i], pts[i + 1], elapsed_sec)

        return None

    # ------------------------------------------------------------------
    # Build velocity reference from planner trajectory
    # q_d(t), qd_d(t), qdd_d(t) are sampled from the global planner output.
    # Then we form a tracking reference for the lower-level torque loop.
    # ------------------------------------------------------------------
    def _compute_trajectory_velocity_command(self) -> Optional[np.ndarray]:
        if not self.traj_tracking_enabled:
            return None
        if not self.traj_is_active:
            return None
        if self.active_traj is None or self.active_traj_start_time is None:
            return None

        elapsed = (self.get_clock().now() - self.active_traj_start_time).nanoseconds * 1e-9
        sampled = self._sample_active_trajectory(elapsed)
        if sampled is None:
            return None

        q_ref, qd_ref, qdd_ref = sampled
        q_meas, qd_meas = self._read_torque_joint_state()

        # Reference tracking law:
        # v_des = qd_ref + Kp_traj * (q_ref - q_meas)
        # This lets the existing torque loop keep doing the low-level work.
        v_des = self.traj_velocity_ff_scale * qd_ref

        if self.traj_use_position_feedback:
            v_des = v_des + self.traj_position_kp * (q_ref - q_meas)

        # Optional small accel feed-forward can be introduced here if desired.
        # For now it is mapped softly into velocity reference.
        if abs(self.traj_acceleration_ff_scale) > 0.0:
            v_des = v_des + self.traj_acceleration_ff_scale * qdd_ref

        self.last_q_ref = q_ref
        self.last_qd_ref = qd_ref
        self.last_qdd_ref = qdd_ref
        self.last_v_des = v_des

        return v_des

    # ------------------------------------------------------------------
    # Sensor publication
    # ------------------------------------------------------------------
    def publish_sensor_data(self):
        now = self.get_clock().now().to_msg()

        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = list(self.joint_names)

        q, qd = [], []
        qd_by_name = {}
        for name in self.joint_names:
            pos_val = self._get_sensor(f"{name}_pos")
            vel_val = self._get_sensor(f"{name}_vel")
            q.extend(pos_val)
            qd.extend(vel_val)
            qd_by_name[name] = float(vel_val[0]) if len(vel_val) > 0 else 0.0

        for i, name in enumerate(self.torque_joint_names):
            v_meas[i] = qd_by_name.get(name, 0.0)

        joint_msg.position = [float(v) for v in q]
        joint_msg.velocity = [float(v) for v in qd]
        joint_msg.effort = [0.0] * len(self.joint_names)
        self.pub_joint_std.publish(joint_msg)
        self.pub_joint_hanyang.publish(joint_msg)

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"

        quat = self._get_sensor("base_quat")
        if len(quat) >= 4:
            imu_msg.orientation.w = float(quat[0])
            imu_msg.orientation.x = float(quat[1])
            imu_msg.orientation.y = float(quat[2])
            imu_msg.orientation.z = float(quat[3])

        gyro = self._get_sensor("base_gyro")
        if len(gyro) >= 3:
            imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = map(float, gyro[:3])

        acc = self._get_sensor("base_acc")
        if len(acc) >= 3:
            imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = map(float, acc[:3])

        self.pub_imu.publish(imu_msg)

        pos_msg = Vector3Stamped()
        pos_msg.header.stamp = now
        pos_msg.header.frame_id = "base_link"
        pos = self._get_sensor("frame_pos")
        if len(pos) >= 3:
            pos_msg.vector.x, pos_msg.vector.y, pos_msg.vector.z = map(float, pos[:3])
        self.pub_pos.publish(pos_msg)

        com_msg = Vector3Stamped()
        com_msg.header.stamp = now
        com_msg.header.frame_id = "world"
        com = self._compute_world_com()
        com_msg.vector.x = float(com[0])
        com_msg.vector.y = float(com[1])
        com_msg.vector.z = float(com[2])
        self.pub_com.publish(com_msg)

        if len(quat) >= 4 and len(pos) >= 3:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "world"
            t.child_frame_id = "base_link"
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            t.transform.rotation.w = float(quat[0])
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            self.tf_broadcaster.sendTransform(t)


def main():
    package_path = get_package_share_directory('forestry_robot_mjcf')
    xml_path = os.path.join(package_path, 'xml', 'scene2_upper.xml')

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    ctrl_array = np.zeros(m.nu)

    mujoco.mj_forward(m, d)

    def qadr(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"joint '{name}' not found")
        return int(m.jnt_qposadr[jid])

    init_upper = {
        "UPJ1": 0.0,
        "UPJ2": 0.4,
        "UPJ3": -1.08,
        "UPJ4": 0.65,
        "UPJ5": 0.3,
        "UPJ6": 0.0,
        "TOOLJ1": 0.0,
    }
    for name, val in init_upper.items():
        d.qpos[qadr(name)] = float(val)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    joint_names = ["UPJ1", "UPJ2", "UPJ3", "UPJ4", "UPJ5", "UPJ6", "TOOLJ1"]
    torque_joint_names = ["UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"]

    def dofadr_from_joint(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"joint '{name}' not found")
        return int(m.jnt_dofadr[jid])

    dof_ids = np.array([dofadr_from_joint(n) for n in torque_joint_names], dtype=int)

    sensor_lookup = {}
    for name in joint_names:
        for suffix in ['_pos', '_vel']:
            sensor_name = name + suffix
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sid != -1:
                sensor_lookup[sensor_name] = (m.sensor_adr[sid], m.sensor_dim[sid])

    for s in ["base_quat", "base_gyro", "base_acc", "frame_pos", "frame_vel"]:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s)
        if sid != -1:
            sensor_lookup[s] = (m.sensor_adr[sid], m.sensor_dim[sid])
        else:
            print(f"[경고] 센서 '{s}'가 MJCF에 정의되어 있지 않습니다.")

    node_holder = {"node": None}

    def ros_spin():
        rclpy.init()
        node = UnifiedUpperNode(m, d, ctrl_array, joint_names, torque_joint_names, sensor_lookup)
        node_holder["node"] = node
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    threading.Thread(target=ros_spin, daemon=True).start()

    # Wait until ROS node is ready
    while node_holder["node"] is None:
        pass

    ros_node: UnifiedUpperNode = node_holder["node"]

    n_torque = len(torque_joint_names)
    int_err = np.zeros(n_torque)

    Kp = np.array([100000, 60000, 200000, 30000, 500], dtype=float)
    Ki = np.array([30000000, 50000000, 1000000, 2500000, 5000], dtype=float)
    Kd = np.array([0, 0, 0, 0, 0], dtype=float)
    torque_limit = np.array([50000000, 50000000, 50000000, 50000000, 1000000], dtype=float)

    alpha = 0.2
    e_prev = np.zeros(n_torque)
    dedt_filt = np.zeros(n_torque)

    AW_MODE = "clamp"
    K_AW = 200.0
    E_INT = 0.01

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            dt = m.opt.timestep

            mujoco.mj_forward(m, d)

            # Gravity / bias feedforward
            tau_ff = d.qfrc_bias[dof_ids].copy()

            # ----------------------------------------------------------
            # High-level reference source selection
            # Priority:
            # 1) Active planner trajectory tracking
            # 2) External /hanyang/velocity_cmd
            # ----------------------------------------------------------
            v_des = ctrl_array[:n_torque].copy()

            traj_v_des = ros_node._compute_trajectory_velocity_command()
            if traj_v_des is not None and ros_node.traj_override_velocity_cmd:
                v_des = traj_v_des.copy()

            # Existing low-level velocity-to-torque loop
            e = v_des - v_meas
            dedt_raw = (e - e_prev) / max(dt, 1e-9)
            dedt_filt = dedt_filt + alpha * (dedt_raw - dedt_filt)
            e_prev = e.copy()

            u_fb = Kp * e + Ki * int_err + Kd * dedt_filt
            u_unsat = tau_ff + u_fb
            u_sat = np.clip(u_unsat, -torque_limit, torque_limit)

            if AW_MODE == "backcalc":
                for i in range(n_torque):
                    if Ki[i] > 0.0:
                        e_eff = e[i] if abs(e[i]) > E_INT else 0.0
                        int_err[i] += (e_eff + K_AW * (u_sat[i] - u_unsat[i]) / Ki[i]) * dt
            else:
                for i in range(n_torque):
                    integrate = True
                    if u_sat[i] != u_unsat[i]:
                        if u_unsat[i] > torque_limit[i]:
                            integrate = (e[i] < 0.0)
                        elif u_unsat[i] < -torque_limit[i]:
                            integrate = (e[i] > 0.0)
                    if integrate and Ki[i] > 0.0 and abs(e[i]) > E_INT:
                        int_err[i] += e[i] * dt

            d.ctrl[:n_torque] = u_sat

            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == '__main__':
    main()