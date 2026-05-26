#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3, Vector3Stamped, TransformStamped
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
    def __init__(self, model, data, ctrl_array, joint_names, sensor_lookup):
        super().__init__('sim_ecvt2_node')

        # ------------ MuJoCo 핸들 ------------
        self.model = model
        self.data = data
        self.ctrl_array = ctrl_array

        # ------------ 로봇 메타데이터 ------------
        self.joint_names = joint_names  # 총 28개
        self.sensor_lookup = sensor_lookup  # sensordata 주소/차원 테이블

        # ============= ROS2 Sub/Pub 초기화 =============
        # 제어 명령 구독 (JointState.velocity 사용)
        self.create_subscription(JointState, '/hanyang/velocity_cmd', self.velocity_callback, 10)

        # JointState 퍼블리셔 두 개 (표준 + 사용자용)
        self.pub_joint_std = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_joint_hanyang = self.create_publisher(JointState, '/hanyang/joint_states', 10)

        # IMU, base_pos 퍼블리셔
        self.pub_imu = self.create_publisher(Imu, '/hanyang/imu', 10)
        self.pub_pos = self.create_publisher(Vector3Stamped, '/hanyang/base_pos', 10)
        self.pub_com = self.create_publisher(Vector3Stamped, '/hanyang/com_pos', 10)

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

    # ============================================================
    # 콜백: 제어 명령 (/hanyang/velocity_cmd)
    # ============================================================
    def velocity_callback(self, msg: JointState):
        # if len(msg.velocity) != len(self.ctrl_array)-4:
        if len(msg.velocity) != len(self.ctrl_array) + 1:
            self.get_logger().warn(f"Expected {len(self.ctrl_array)-4} velocities, got {len(msg.velocity)}")
            return
        # MuJoCo 제어 입력 갱신
        self.ctrl_array[:25] = msg.velocity[:25]

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

    def _compute_world_com(self):
        # d.xipos: 각 body의 world COM 위치, m.body_mass: 각 body 질량
        masses = np.asarray(self.model.body_mass, dtype=float)
        xipos = np.asarray(self.data.xipos, dtype=float)
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return np.zeros(3, dtype=float)
        weighted = masses[:, None] * xipos
        return np.sum(weighted, axis=0) / total_mass

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

        # ---------------- COM Position (world) ----------------
        com_msg = Vector3Stamped()
        com_msg.header.stamp = now
        com_msg.header.frame_id = "world"
        com = self._compute_world_com()
        com_msg.vector.x = float(com[0])
        com_msg.vector.y = float(com[1])
        com_msg.vector.z = float(com[2])
        self.pub_com.publish(com_msg)

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
        "UPJ1": 0.0, "UPJ2": 0.4, "UPJ3": -1.08, "UPJ4": 0.65,
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
    # d.qpos[0:7] = np.array([0.0, -0.002, 3.4, 1.0, 0.0, 0.0, 0.0], dtype=float)
    yaw_deg = 0.0  # 원하는 각도(deg) 넣기
    yaw_rad = np.deg2rad(yaw_deg)

    qw = np.cos(yaw_rad * 0.5)
    qz = np.sin(yaw_rad * 0.5)

    d.qpos[0:7] = np.array([0.0, -0.002, 3.4, qw, 0.0, 0.0, qz], dtype=float)


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
    def ros_spin():
        rclpy.init()
        node = UnifiedRobotNode(m, d, ctrl_array, joint_names, sensor_lookup)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    threading.Thread(target=ros_spin, daemon=True).start()

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
            v_des = ctrl_array[:n_torque].copy()

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
