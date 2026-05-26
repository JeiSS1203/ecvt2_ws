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
from geometry_msgs.msg import Vector3Stamped, TransformStamped
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

        self.create_subscription(JointState, '/hanyang/velocity_cmd', self.velocity_callback, 10)

        self.pub_joint_std = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_joint_hanyang = self.create_publisher(JointState, '/hanyang/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/hanyang/imu', 10)
        self.pub_pos = self.create_publisher(Vector3Stamped, '/hanyang/base_pos', 10)
        self.pub_com = self.create_publisher(Vector3Stamped, '/hanyang/com_pos', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.01, self.publish_sensor_data)

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

    def ros_spin():
        rclpy.init()
        node = UnifiedUpperNode(m, d, ctrl_array, joint_names, torque_joint_names, sensor_lookup)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    threading.Thread(target=ros_spin, daemon=True).start()

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
            tau_ff = d.qfrc_bias[dof_ids].copy()

            v_des = ctrl_array[:n_torque].copy()
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
