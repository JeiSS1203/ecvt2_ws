#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import threading
from typing import Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3, Vector3Stamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory


v_meas = np.zeros(25, dtype=float)
# ============================================================
# UnifiedRobotNode : MuJoCo 시뮬레이션 → ROS2 퍼블리셔 노드
#   - /joint_states        : robot_state_publisher 용
#   - /hanyang/joint_states: 사용자 제어기 용
#   - /hanyang/imu         : IMU (orientation, gyro, accel)
#   - /hanyang/base_pos    : 로봇의 위치 벡터 (world 기준 위치가 필요하면 TF 사용)
#   - TF : world → base_link
# ============================================================
class UnifiedRobotNode(Node):
    def __init__(self, model, data, ctrl_array, joint_names, torque_joint_names, sensor_lookup):
        super().__init__('sim_ecvt2_node')

        # ------------ MuJoCo 핸들 ------------
        self.model = model
        self.data = data
        self.ctrl_array = ctrl_array

        # ------------ 로봇 메타데이터 ------------
        self.joint_names = joint_names  # 총 28개
        self.torque_joint_names = torque_joint_names
        self.sensor_lookup = sensor_lookup  # sensordata 주소/차원 테이블

        # ------------ Planner trajectory tracking parameters ------------
        self.declare_parameter('trajectory_topic', '/vp_sto_global_planner_full_node/actuated_reference')
        self.declare_parameter('traj_tracking_enabled', True)
        self.declare_parameter('traj_hold_last_point', True)
        self.declare_parameter('traj_use_position_feedback', True)
        self.declare_parameter('traj_position_kp', 1.5)
        self.declare_parameter('traj_velocity_ff_scale', 1.0)
        self.declare_parameter('traj_acceleration_ff_scale', 0.0)
        self.declare_parameter('traj_override_velocity_cmd', True)
        self.declare_parameter('passive_csv_enabled', True)
        self.declare_parameter('passive_csv_path', '/home/jin/harco/ecvt2_ws/upj5_upj6_joint_states.csv')

        self.trajectory_topic = self.get_parameter('trajectory_topic').get_parameter_value().string_value
        self.traj_tracking_enabled = self.get_parameter('traj_tracking_enabled').get_parameter_value().bool_value
        self.traj_hold_last_point = self.get_parameter('traj_hold_last_point').get_parameter_value().bool_value
        self.traj_use_position_feedback = self.get_parameter('traj_use_position_feedback').get_parameter_value().bool_value
        self.traj_position_kp = self.get_parameter('traj_position_kp').get_parameter_value().double_value
        self.traj_velocity_ff_scale = self.get_parameter('traj_velocity_ff_scale').get_parameter_value().double_value
        self.traj_acceleration_ff_scale = self.get_parameter('traj_acceleration_ff_scale').get_parameter_value().double_value
        self.traj_override_velocity_cmd = self.get_parameter('traj_override_velocity_cmd').get_parameter_value().bool_value
        self.passive_csv_enabled = self.get_parameter('passive_csv_enabled').get_parameter_value().bool_value
        self.passive_csv_path = self.get_parameter('passive_csv_path').get_parameter_value().string_value

        self.active_traj: Optional[JointTrajectory] = None
        self.traj_is_active = False
        self.traj_elapsed_sim = 0.0
        self.last_q_ref = None
        self.last_qd_ref = None
        self.last_qdd_ref = None
        self.last_v_des = None
        self.last_elapsed_sim = 0.0
        self.passive_csv_file = None
        self.passive_csv_writer = None
        self.passive_csv_recording = False
        self.passive_csv_completed = False
        self.passive_csv_prev_vel = None
        self._state_lock = threading.Lock()

        # ============= ROS2 Sub/Pub 초기화 =============
        # 제어 명령 구독 (JointState.velocity 사용)
        self.create_subscription(JointState, '/hanyang/velocity_cmd', self.velocity_callback, 10)
        self.create_subscription(JointTrajectory, self.trajectory_topic, self.trajectory_callback, 10)

        # JointState 퍼블리셔 두 개 (표준 + 사용자용)
        self.pub_joint_std = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_joint_hanyang = self.create_publisher(JointState, '/hanyang/joint_states', 10)

        # IMU, base_pos 퍼블리셔
        self.pub_imu = self.create_publisher(Imu, '/hanyang/imu', 10)
        self.pub_pos = self.create_publisher(Vector3Stamped, '/hanyang/base_pos', 10)

        # TF 브로드캐스터 (world → base_link)
        self.tf_broadcaster = TransformBroadcaster(self)

         # EE 타겟 퍼블리셔
        self.pub_target = self.create_publisher(Vector3, '/hanyang/ee_target', 10)
        # ---- target 식별자 보관 ----
        self.sid_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_id = -1
        if bid != -1:
            self.target_mocap_id = int(self.model.body_mocapid[bid])

        if self.sid_target == -1 and self.target_mocap_id == -1:
            self.get_logger().warn("target_site와 mocap body 'target'을 찾지 못했습니다.")
        else:
            self.get_logger().info(f"[Target IDs] sid_target={self.sid_target}, mocap_id={self.target_mocap_id}")

        # 타이머: 100Hz 센서 퍼블리시
        self.create_timer(0.01, self.publish_sensor_data)

    def destroy_node(self):
        self._close_passive_csv()
        super().destroy_node()

    # ============================================================
    # 콜백: 제어 명령 (/hanyang/velocity_cmd)
    # ============================================================
    def velocity_callback(self, msg: JointState):
        n_ctrl = len(self.torque_joint_names)

        if len(msg.name) == len(msg.velocity) and len(msg.name) > 0:
            name_to_vel = dict(zip(msg.name, msg.velocity))
            matched = False
            with self._state_lock:
                for i, joint_name in enumerate(self.torque_joint_names):
                    if joint_name in name_to_vel:
                        self.ctrl_array[i] = name_to_vel[joint_name]
                        matched = True
            if matched:
                return

        if len(msg.velocity) < n_ctrl:
            self.get_logger().warn(f"Expected at least {n_ctrl} velocities, got {len(msg.velocity)}")
            return

        with self._state_lock:
            self.ctrl_array[:n_ctrl] = msg.velocity[:n_ctrl]

    # ============================================================
    # 콜백: planner trajectory
    # - /actuated_reference: torque_joint_names 25개를 그대로 받음
    # - /whole_body_reference: base_* 6개 + actuated joints에서 actuated 부분만 추출
    # ============================================================
    def trajectory_callback(self, msg: JointTrajectory):
        if not self.traj_tracking_enabled:
            return

        if len(msg.points) == 0:
            self.get_logger().warn("Received empty JointTrajectory")
            return

        expected = list(self.torque_joint_names)
        got = list(msg.joint_names)
        if got == expected:
            tracking_msg = msg
        else:
            name_to_index = {name: i for i, name in enumerate(got)}
            missing = [name for name in expected if name not in name_to_index]
            if missing:
                self.get_logger().warn(
                    f"Trajectory joint names mismatch. missing={missing}, expected={expected}, got={got}"
                )
                return

            tracking_msg = JointTrajectory()
            tracking_msg.header = msg.header
            tracking_msg.joint_names = expected
            indices = [name_to_index[name] for name in expected]

            for src_pt in msg.points:
                dst_pt = JointTrajectoryPoint()
                dst_pt.time_from_start = src_pt.time_from_start
                dst_pt.positions = [
                    float(src_pt.positions[i]) for i in indices
                ] if len(src_pt.positions) == len(got) else []
                dst_pt.velocities = [
                    float(src_pt.velocities[i]) for i in indices
                ] if len(src_pt.velocities) == len(got) else []
                dst_pt.accelerations = [
                    float(src_pt.accelerations[i]) for i in indices
                ] if len(src_pt.accelerations) == len(got) else []
                tracking_msg.points.append(dst_pt)

            if len(tracking_msg.points) == 0:
                self.get_logger().warn("Filtered whole-body trajectory has no points")
                return

        if list(tracking_msg.joint_names) != expected:
            self.get_logger().warn(
                f"Filtered trajectory joint names mismatch. expected={expected}, got={tracking_msg.joint_names}"
            )
            return

        with self._state_lock:
            self.active_traj = tracking_msg
            self.traj_is_active = True
            self.traj_elapsed_sim = 0.0
            self.passive_csv_prev_vel = None
            self.passive_csv_completed = False
            self._close_passive_csv()

        t_last = self._point_time_sec(tracking_msg.points[-1])
        self.get_logger().info(
            f"Received planner trajectory on {self.trajectory_topic}: {len(tracking_msg.points)} points, duration={t_last:.3f}s"
        )

    # ============================================================
    # 센서 데이터 접근 함수 (MuJoCo sensordata → numpy array)
    # ============================================================
    def _get_sensor(self, name: str):
        if name not in self.sensor_lookup:
            # 센서가 정의되어 있지 않으면 0 반환 (경고 1회/주기)
            self.get_logger().warn(f"Sensor '{name}' not found")
            return [0.0]
        addr, dim = self.sensor_lookup[name]
        return self.data.sensordata[addr:addr + dim]

    def _open_passive_csv_if_needed(self):
        if not self.passive_csv_enabled or self.passive_csv_writer is not None:
            return

        csv_dir = os.path.dirname(self.passive_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        self.passive_csv_file = open(self.passive_csv_path, 'w', newline='')
        self.passive_csv_writer = csv.writer(self.passive_csv_file)
        self.passive_csv_writer.writerow([
            'ros_time_sec',
            'sim_time_sec',
            'trajectory_elapsed_sec',
            'UPJ5_position',
            'UPJ5_velocity',
            'UPJ5_acceleration',
            'UPJ6_position',
            'UPJ6_velocity',
            'UPJ6_acceleration',
        ])
        self.passive_csv_file.flush()
        self.passive_csv_recording = True
        self.get_logger().info(f"Recording passive joint CSV to: {self.passive_csv_path}")

    def _close_passive_csv(self):
        if self.passive_csv_file is not None:
            self.passive_csv_file.flush()
            self.passive_csv_file.close()
            self.passive_csv_file = None
            self.passive_csv_writer = None
            self.passive_csv_recording = False

    def record_passive_sample(self, dt: float):
        if self.passive_csv_completed:
            return
        if not self.passive_csv_enabled or self.active_traj is None or len(self.active_traj.points) == 0:
            return

        self._open_passive_csv_if_needed()
        if self.passive_csv_writer is None:
            return

        upj5_pos = self._get_sensor('UPJ5_pos')
        upj5_vel = self._get_sensor('UPJ5_vel')
        upj6_pos = self._get_sensor('UPJ6_pos')
        upj6_vel = self._get_sensor('UPJ6_vel')

        vel = np.array([
            float(upj5_vel[0]) if len(upj5_vel) > 0 else 0.0,
            float(upj6_vel[0]) if len(upj6_vel) > 0 else 0.0,
        ], dtype=float)
        if self.passive_csv_prev_vel is None:
            acc = np.zeros(2, dtype=float)
        else:
            acc = (vel - self.passive_csv_prev_vel) / max(float(dt), 1e-9)
        self.passive_csv_prev_vel = vel.copy()

        now = self.get_clock().now().to_msg()
        self.passive_csv_writer.writerow([
            float(now.sec) + float(now.nanosec) * 1e-9,
            float(self.data.time),
            float(self.traj_elapsed_sim),
            float(upj5_pos[0]) if len(upj5_pos) > 0 else 0.0,
            float(vel[0]),
            float(acc[0]),
            float(upj6_pos[0]) if len(upj6_pos) > 0 else 0.0,
            float(vel[1]),
            float(acc[1]),
        ])

        t_last = self._point_time_sec(self.active_traj.points[-1])
        if self.traj_elapsed_sim >= t_last:
            self._close_passive_csv()
            self.passive_csv_completed = True
            self.get_logger().info(f"Saved passive joint CSV: {self.passive_csv_path}")

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

        if len(p0.velocities) == n and len(p1.velocities) == n:
            v0 = np.array(p0.velocities, dtype=float)
            v1 = np.array(p1.velocities, dtype=float)
        else:
            slope = (q1 - q0) / dt
            v0 = slope.copy()
            v1 = slope.copy()

        h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
        h10 = tau**3 - 2.0 * tau**2 + tau
        h01 = -2.0 * tau**3 + 3.0 * tau**2
        h11 = tau**3 - tau**2
        q = h00 * q0 + h10 * dt * v0 + h01 * q1 + h11 * dt * v1

        dh00 = 6.0 * tau**2 - 6.0 * tau
        dh10 = 3.0 * tau**2 - 4.0 * tau + 1.0
        dh01 = -6.0 * tau**2 + 6.0 * tau
        dh11 = 3.0 * tau**2 - 2.0 * tau
        qd = (dh00 * q0 + dh10 * dt * v0 + dh01 * q1 + dh11 * dt * v1) / dt

        d2h00 = 12.0 * tau - 6.0
        d2h10 = 6.0 * tau - 4.0
        d2h01 = -12.0 * tau + 6.0
        d2h11 = 6.0 * tau - 2.0
        qdd = (d2h00 * q0 + d2h10 * dt * v0 + d2h01 * q1 + d2h11 * dt * v1) / (dt * dt)

        return q, qd, qdd

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
            self.traj_is_active = False
            return None

        for i in range(len(pts) - 1):
            t0 = self._point_time_sec(pts[i])
            t1 = self._point_time_sec(pts[i + 1])
            if t0 <= elapsed_sec <= t1:
                return self._interpolate_segment(pts[i], pts[i + 1], elapsed_sec)

        return None

    def _compute_trajectory_velocity_command(self) -> Optional[np.ndarray]:
        if not self.traj_tracking_enabled or not self.traj_is_active or self.active_traj is None:
            return None

        elapsed = self.traj_elapsed_sim
        self.last_elapsed_sim = elapsed

        sampled = self._sample_active_trajectory(elapsed)
        if sampled is None:
            return None

        q_ref, qd_ref, qdd_ref = sampled
        q_meas, _ = self._read_torque_joint_state()

        v_des = self.traj_velocity_ff_scale * qd_ref
        if self.traj_use_position_feedback:
            v_des = v_des + self.traj_position_kp * (q_ref - q_meas)
        if abs(self.traj_acceleration_ff_scale) > 0.0:
            v_des = v_des + self.traj_acceleration_ff_scale * qdd_ref

        self.last_q_ref = q_ref
        self.last_qd_ref = qd_ref
        self.last_qdd_ref = qdd_ref
        self.last_v_des = v_des
        return v_des

    def advance_traj_time(self, dt: float):
        if not self.traj_is_active or self.active_traj is None or len(self.active_traj.points) == 0:
            return

        self.traj_elapsed_sim += float(dt)

        t_last = self._point_time_sec(self.active_traj.points[-1])
        if self.traj_elapsed_sim > t_last:
            if self.traj_hold_last_point:
                self.traj_elapsed_sim = t_last
            else:
                self.traj_is_active = False
    
    # ============================================================
    # EE 타겟 위치 퍼블리시
    # ============================================================
    def publish_target(self):
        msg = Vector3()
        # 현재 target mocap 위치를 그대로 퍼블리시
        if hasattr(self, "target_mocap_id") and self.target_mocap_id != -1:
            pos = self.data.mocap_pos[self.target_mocap_id]
            msg.x, msg.y, msg.z = map(float, pos[:3])
        self.pub_target.publish(msg)

    # ============================================================
    # 주기적으로 센서 데이터 퍼블리시 (JointState, IMU, TF 등)
    # ============================================================
    def publish_sensor_data(self):
        now = self.get_clock().now().to_msg()

        # ---------------- JointState ----------------
        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = list(self.joint_names)  # 모든 조인트 포함

        q, qd = [], []
        for name in self.joint_names:
            q.extend(self._get_sensor(f"{name}_pos"))
            qd.extend(self._get_sensor(f"{name}_vel"))
            
        # joint_names 순서대로 돌면서 'd_joint1', 'd_joint2' 제외. 제어용
        j = 0  # v_meas에 채워 넣을 인덱스
        for i, name in enumerate(self.joint_names):
            if name == "UPJ5" or name == "UPJ6":
                continue  # 이 두 조인트는 건너뛰기
            v_meas[j] = qd[i]
            j += 1

        joint_msg.position = [float(v) for v in q]
        joint_msg.velocity = [float(v) for v in qd]
        joint_msg.effort = [0.0] * len(self.joint_names)

        # 두 토픽에 동시에 퍼블리시
        self.pub_joint_std.publish(joint_msg)
        self.pub_joint_hanyang.publish(joint_msg)

        # ---------------- IMU ----------------
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "base_link"

        quat = self._get_sensor("base_quat")  # MuJoCo quaternion [w, x, y, z]
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

        # ---------------- Base Position ----------------
        pos_msg = Vector3Stamped()
        pos_msg.header.stamp = now
        pos_msg.header.frame_id = "base_link"
        pos = self._get_sensor("frame_pos")
        if len(pos) >= 3:
            pos_msg.vector.x, pos_msg.vector.y, pos_msg.vector.z = map(float, pos[:3])
        self.pub_pos.publish(pos_msg)

        # ---------------- TF (world → base_link) ----------------
        # RViz에서 Fixed Frame을 'world'로 두고, URDF base_link에 모델을 놓고 싶을 때 사용
        if len(quat) >= 4 and len(pos) >= 3:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "world"
            t.child_frame_id = "base_link"
            # 회전: TF는 [x, y, z, w] 순서
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            t.transform.rotation.w = float(quat[0])
            # 위치
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            self.tf_broadcaster.sendTransform(t)
        
        # ---------------- EE 타겟 퍼블리시 ----------------
        # self.publish_target()


# ============================================================
# main 함수
#   - MuJoCo 모델 로드
#   - ROS2 노드 실행 (Thread)
#   - MuJoCo Viewer 실행
# ============================================================
def main():
    # ============= 모델 경로 =============
    package_path = get_package_share_directory('forestry_robot_mjcf')
    xml_path = os.path.join(package_path, 'xml', 'scene2.xml')

    # ============= MuJoCo 모델 로딩 =============
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    ctrl_array = np.zeros(m.nu)

    # --- (중요) target mocap 초기화 ---
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    if bid == -1:
        print("[경고] body 'target'을 찾지 못했습니다. XML에 <body name=\"target\" ...> 필요")
    else:
        mocapid = int(m.body_mocapid[bid])
        if mocapid == -1:
            print("[경고] 'target'은 mocap 바디가 아님. <body name=\"target\" mocap=\"true\"> 로 변경")
        else:
            d.mocap_pos[mocapid]  = m.body_pos[bid].copy()
            if m.body_quat.size >= (bid+1)*4:
                d.mocap_quat[mocapid] = m.body_quat[bid].copy()
            else:
                d.mocap_quat[mocapid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # --- 파생값 계산(첫 프레임부터 유효 좌표 보장) ---
    mujoco.mj_forward(m, d)
     # [중요] 여기서 쓰는 헬퍼는 _qadr 로 이름을 달리해서
    # 뒤쪽에 이미 있는 def jadr(...)와 이름 충돌을 피한다.
    def _qadr(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"joint '{name}' not found")
        return int(m.jnt_qposadr[jid])

    # 1) 네가 원하는 기본 관절 각도(라디안)만 지정
    init_base = {
        # # === FR ===
        # "FRJ1": 0.0, "FRJ2": 0.0, "FRJ3": 0.0, "FRJ5": 0.0, "FRJW": 0.0,
        # # === FL ===
        # "FLJ1": 0.0, "FLJ2": 0.0, "FLJ3": 0.0, "FLJ5": 0.0, "FLJW": 0.0,
        # # === RR ===
        # "RRJ1": 0.0, "RRJ2": 0.0, "RRJ3": 0.0, "RRJ5": 0.0, "RRJW": 0.0,
        # # === RL ===
        # "RLJ1": 0.0, "RLJ2": 0.0, "RLJ3": 0.0, "RLJ5": 0.0, "RLJW": 0.0,

        # === FR ===
        "FRJ1": -np.deg2rad(20), "FRJ2": np.deg2rad(20.925), "FRJ3": np.deg2rad(-20.925), "FRJ5": np.deg2rad(20), "FRJW": 0.0, # 0.5m 높이
        # === FL ===
        "FLJ1": np.deg2rad(20), "FLJ2": np.deg2rad(50.0), "FLJ3": np.deg2rad(-20.746 ), "FLJ5": -np.deg2rad(20), "FLJW": 0.0, # 1.2m 높이
        # === RR ===
        "RRJ1": np.deg2rad(20), "RRJ2": np.deg2rad(-45.585), "RRJ3": np.deg2rad(45.585 ), "RRJ5": -np.deg2rad(20), "RRJW": 0.0, # 1.0m 높이
        # === RL ===
        "RLJ1": -np.deg2rad(20), "RLJ2": np.deg2rad(-30.0), "RLJ3": np.deg2rad( 30.0 ), "RLJ5": np.deg2rad(20), "RLJW": 0.0, # 0.7m 높이


        # === 붐/툴 ===
        "UPJ1": np.deg2rad(70.0), "UPJ2": 0.4, "UPJ3": -1.08, "UPJ4": 0.65,
        "UPJ5": 0.3, "UPJ6": 0.0, "TOOLJ1": 0.0,
    }

    # 2) 종속 조인트는 제약에 맞춰 자동 계산
    FRJ2, FRJ3 = init_base["FRJ2"], init_base["FRJ3"]
    FLJ2, FLJ3 = init_base["FLJ2"], init_base["FLJ3"]
    RRJ2, RRJ3 = init_base["RRJ2"], init_base["RRJ3"]
    RLJ2, RLJ3 = init_base["RLJ2"], init_base["RLJ3"]

    init_dep = {
        # 공통 텐던: J4_2 = -(J2 + J3)
        "FRJ4": -(FRJ2 + FRJ3),
        "FLJ4": -(FLJ2 + FLJ3),
        "RRJ4": -(RRJ2 + RRJ3),
        "RLJ4": -(RLJ2 + RLJ3),
        # BJ3 mimic (앞다리 -, 뒷다리 +)
        "FRDJ2": -(FRJ2 + FRJ3),
        "FLDJ2": -(FLJ2 + FLJ3),
        "RRDJ2":  (RRJ2 + RRJ3),
        "RLDJ2":  (RLJ2 + RLJ3),
    }

    # 3) freejoint(base) 초기값 (원하면 조정)
    # d.qpos[0:7] = np.array([0.0, -0.002, 2.4, 1.0, 0.0, 0.0, 0.0], dtype=float)
    d.qpos[0:7] = np.array([0.0, -0.002, 3.4, 1.0, 0.0, 0.0, 0.0], dtype=float)


    # 4) qpos에 기록 (이름→주소 매핑 사용)
    for name, val in {**init_base, **init_dep}.items():
        d.qpos[_qadr(name)] = float(val)

    # 5) 속도 0, 파생치 갱신, (선택) qpos0 동기화
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # ---- helper: joint qpos addr, actuator id
    def jadr(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"joint '{name}' not found")
        return m.jnt_qposadr[jid]

    def aid(name: str) -> int:
        aid_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid_ == -1:
            raise RuntimeError(f"actuator '{name}' not found")
        return aid_

    # ---- qpos 인덱스 (각 다리의 2,3번 조인트)
    q_FR2 = jadr("FRJ2"); q_FR3 = jadr("FRJ3")
    q_FL2 = jadr("FLJ2"); q_FL3 = jadr("FLJ3")
    q_RR2 = jadr("RRJ2"); q_RR3 = jadr("RRJ3")
    q_RL2 = jadr("RLJ2"); q_RL3 = jadr("RLJ3")

    # # ---- pre 서보 액추에이터 인덱스
    # a_FR_pre = aid("FRJ4_pre_servo")
    # a_FL_pre = aid("FLJ4_pre_servo")
    # a_RR_pre = aid("RRJ4_pre_servo")
    # a_RL_pre = aid("RLJ4_pre_servo")

    # ============= 로봇 조인트 이름 (총 27개) =============
    joint_names = [
        "FRJ1", "FRJ2", "FRJ3", "FRJ5", "FRJW",
        "FLJ1", "FLJ2", "FLJ3", "FLJ5", "FLJW",
        "RRJ1", "RRJ2", "RRJ3", "RRJ5", "RRJW",
        "RLJ1", "RLJ2", "RLJ3", "RLJ5", "RLJW",
        "UPJ1", "UPJ2", "UPJ3", "UPJ4",
        "UPJ5", "UPJ6", "TOOLJ1"
    ]

    # ===== (추가) 토크 제어 대상 25개 조인트를 명시 =====
    torque_joint_names = [
        "FRJ1","FRJ2","FRJ3","FRJ5","FRJW",
        "FLJ1","FLJ2","FLJ3","FLJ5","FLJW",
        "RRJ1","RRJ2","RRJ3","RRJ5","RRJW",
        "RLJ1","RLJ2","RLJ3","RLJ5","RLJW",
        "UPJ1","UPJ2","UPJ3","UPJ4","TOOLJ1"
    ]  # 총 25개

    def dofadr_from_joint(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"joint '{name}' not found")
        return int(m.jnt_dofadr[jid])  # hinge/slide 1DOF면 이 값이 해당 dof 인덱스

    dof_ids = np.array([dofadr_from_joint(n) for n in torque_joint_names], dtype=int)

    # ============= 센서 인덱스 테이블 구성 =============
    #  각 센서 이름 → (시작주소, 차원) 매핑
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

    # ============= ROS2 스레드 실행 =============
    node_holder = {"node": None}

    def ros_spin():
        rclpy.init()
        node = UnifiedRobotNode(m, d, ctrl_array, joint_names, torque_joint_names, sensor_lookup)
        node_holder["node"] = node
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    threading.Thread(target=ros_spin, daemon=True).start()

    while node_holder["node"] is None:
        pass

    ros_node: UnifiedRobotNode = node_holder["node"]

# ===== PI 제어용 파라미터 =====
    n_torque = 25  # 앞 25개 조인트
    int_err = np.zeros(n_torque)

    # 관절별 Kp, Ki, 최대 토크 설정 (순서는 joint_names와 동일)
    Kp = np.array([
        50000, 50000, 30000, 17000, 0,     # FRJ1 ~ FRJ5
        50000, 50000, 30000, 17000, 0,     # FLJ1 ~ FLJ5
        50000, 50000, 30000, 17000, 0,     # RRJ1 ~ RRJ5
        50000, 50000, 30000, 17000, 0,     # RLJ1 ~ RLJ5
        100000, 60000, 200000, 30000, 500   # UP1,UP2,UP3,UP4,TOOL
    ], dtype=float)

    Ki = np.array([
        5000000, 5000000, 5000000, 3000000, 0,
        5000000, 5000000, 5000000, 3000000, 0,
        5000000, 5000000, 5000000, 3000000, 0,
        5000000, 5000000, 5000000, 3000000, 0,
        30000000, 50000000, 1000000, 2500000, 5000
    ], dtype=float)

    Kd = np.array([
        0, 0, 0, 0, 0,            # FRJ1~FRJ5
        0, 0, 0, 0, 0,            # FLJ1~FRJ5
        0, 0, 0, 0, 0,            # RRJ1~RRJ5
        0, 0, 0, 0, 0,            # RLJ1~RLJ5
        0, 0, 0, 0, 0             # UP1,UP2,UP3,UP4,TOOL
    ], dtype=float)

    # D 항 로패스 필터 비율 (0~1 사이 값)
    # 1.0 → 필터 없음(순수 미분), 0.0 → 완전 평활화
    alpha = 0.2    # 0.1~0.3 정도에서 시작 후 튜닝
    e_prev     = np.zeros(25)   # 이전 오차
    dedt_filt  = np.zeros(25)   # 필터 통과한 오차(관절 속도 오차) 미분

    torque_limit = np.array([
        7000000, 7000000, 7000000, 500000, 800000,
        7000000, 7000000, 7000000, 500000, 800000,
        7000000, 7000000, 7000000, 500000, 800000,
        7000000, 7000000, 7000000, 500000, 800000,
        50000000, 50000000, 50000000, 50000000, 1000000
    ], dtype=float)

    # ===== (추가) Anti-windup 설정 =====
    AW_MODE = "clamp"      # "clamp" 또는 "backcalc"
    K_AW    = 200.0          # back-calculation gain (0.1~1.0 정도부터 튜닝 권장)
    EPS_SAT_PRINT = 1e-6  # 수치잡음 무시 임계


    # ===== MuJoCo Viewer 실행 =====
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            dt = m.opt.timestep

            # ===== (추가) bias(중력 포함) 토크 feedforward 계산 =====
            # qfrc_bias가 최신이 되도록 forward 한 번 호출
            mujoco.mj_forward(m, d)

            # dof-space bias 토크에서 우리가 제어할 25축만 뽑기
            tau_ff = d.qfrc_bias[dof_ids].copy()   # shape (25,)

            # ---- PI 속도 제어
            # v_des = ctrl_array[:n_torque]  # 목표 속도
            with ros_node._state_lock:
                v_des = ctrl_array[:n_torque].copy()
                traj_v_des = ros_node._compute_trajectory_velocity_command()
                ros_node.record_passive_sample(dt)
                ros_node.advance_traj_time(dt)

            if traj_v_des is not None and ros_node.traj_override_velocity_cmd:
                v_des = traj_v_des.copy()

            e = v_des - v_meas
            dedt_raw = (e - e_prev) / max(dt, 1e-9)
            dedt_filt = dedt_filt + alpha * (dedt_raw - dedt_filt)  # alpha는 고정값
            e_prev = e.copy()

            u_fb = Kp * e + Ki * int_err + Kd * dedt_filt
            u_unsat = tau_ff + u_fb

            # u_unsat = Kp * e + Ki * int_err + Kd * dedt_filt
            u_sat = np.clip(u_unsat, -torque_limit, torque_limit)

            for i in range(n_torque):
                sat = abs(u_sat[i] - u_unsat[i]) > EPS_SAT_PRINT
                if i==24 or i==25:
                    name = joint_names[i+2] if i < len(joint_names) else f"J{i+2}"
                else:
                    name = joint_names[i] if i < len(joint_names) else f"J{i}"
                if sat:
                    # 포화 방향 표시(상한/하한/클립)
                    if u_unsat[i] >  torque_limit[i]:
                        direction = "상한"
                    elif u_unsat[i] < -torque_limit[i]:
                        direction = "하한"
                    else:
                        direction = "클립"
                    print(f"[SAT] {i:02d}:{name} dir={direction}  unsat={u_unsat[i]:.3f} -> sat={u_sat[i]:.3f}")



            # =========================
            # Anti-windup 선택형 구현
            # =========================
            E_INT = 0.01  # [rad/s] 또는 해당 축의 속도 단위

            if AW_MODE == "clamp":
                # --- Clamping + deadband
                for i in range(n_torque):
                    integrate = True
                    if u_sat[i] != u_unsat[i]:  # 포화
                        if   u_unsat[i] >  torque_limit[i]: integrate = (e[i] < 0.0)
                        elif u_unsat[i] < -torque_limit[i]: integrate = (e[i] > 0.0)
                    if integrate and Ki[i] > 0.0 and abs(e[i]) > E_INT:
                        int_err[i] += e[i] * dt

            elif AW_MODE == "backcalc":
                # --- Back-Calculation: int_err_dot = e_eff + K_AW*(u_sat - u_unsat)/Ki
                # e_eff만 deadband 적용, backcalc 항은 그대로 둔다(언윈드 보장)
                for i in range(n_torque):
                    if Ki[i] > 0.0:
                        e_eff = e[i] if abs(e[i]) > E_INT else 0.0
                        if i==24 or i==25:
                            K_AW = 0
                        else:
                            K_AW = 200
                        int_err[i] += (e_eff + K_AW * (u_sat[i] - u_unsat[i]) / Ki[i]) * dt

            else:
                # 기본 clamping + deadband
                for i in range(n_torque):
                    integrate = True
                    if u_sat[i] != u_unsat[i]:
                        if   u_unsat[i] >  torque_limit[i]: integrate = (e[i] < 0.0)
                        elif u_unsat[i] < -torque_limit[i]: integrate = (e[i] > 0.0)
                    if integrate and Ki[i] > 0.0 and abs(e[i]) > E_INT:
                        int_err[i] += e[i] * dt

            # 최종 토크 명령 적용
            d.ctrl[:n_torque] = u_sat

            # J5는 Position 제어
            d.ctrl[4]  = 0.0
            d.ctrl[9]  = 0.0
            d.ctrl[14] = 0.0
            d.ctrl[19] = 0.0

            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == '__main__':
    main()
