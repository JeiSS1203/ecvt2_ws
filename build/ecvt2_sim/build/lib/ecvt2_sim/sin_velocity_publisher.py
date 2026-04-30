import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class SinVelocityPublisher(Node):
    def __init__(self):
        super().__init__('sin_velocity_publisher')

        self.publisher_ = self.create_publisher(JointState, '/sim_ecvt/velocity_cmd', 10)

        self.joint_names = [
            "FRJ1", "FRJ2", "FRJ3", "FRJ4", "FRJ5",
            "FLJ1", "FLJ2", "FLJ3", "FLJ4", "FLJ5",
            "RRJ1", "RRJ2", "RRJ3", "RRJ4", "RRJ5",
            "RLJ1", "RLJ2", "RLJ3", "RLJ4", "RLJ5",
            "turntable", "boom", "s_boom", "s_arm", "tool_s", "tool"
        ]
        self.num_joints = len(self.joint_names)

        # 바꿔서 테스트할 수 있는 인덱스
        self.target_joint_index = 21  # 예: 0 → "FRJ1"

        self.freq = 0.5  # Hz
        self.amp = 1   # rad/s
        self.t = 0.0
        self.dt = 0.01
        self.create_timer(self.dt, self.publish_velocity)

    def publish_velocity(self):
        msg = JointState()
        msg.name = self.joint_names
        velocities = [0.0] * self.num_joints
        velocities[self.target_joint_index] = self.amp * math.sin(2 * math.pi * self.freq * self.t)
        msg.velocity = velocities
        self.publisher_.publish(msg)
        self.t += self.dt
    
    def __init__(self):
        super().__init__('sin_velocity_publisher')

        self.publisher_ = self.create_publisher(JointState, '/hanyang/velocity_cmd', 10)

        self.joint_names = [
            "FRJ1", "FRJ2", "FRJ3", "FRJ5", "FRJW",
            "FLJ1", "FLJ2", "FLJ3", "FLJ5", "FLJW",
            "RRJ1", "RRJ2", "RRJ3", "RRJ5", "RRJW",
            "RLJ1", "RLJ2", "RLJ3", "RLJ5", "RLJW",
            "UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"
        ]
        self.num_joints = len(self.joint_names) + 1

        self.freq = 0.25  # Hz
        self.amp = 1.0   # rad/s
        self.t = 0.0
        self.dt = 0.01
        self.create_timer(self.dt, self.publish_velocity)

    def publish_velocity(self):
        msg = JointState()
        msg.name = self.joint_names
        vel_value = self.amp * math.sin(2 * math.pi * self.freq * self.t)
        msg.velocity = [vel_value] * self.num_joints
        msg.velocity[22] = -msg.velocity[22]


        for i in range(20):
            msg.velocity[i] = 0.0  # FRJ1~FRJ5, FLJ1~FLJ5, RRJ1~RRJ5, RLJ1~RLJ5
        # msg.velocity[20] = 0.0  # UPJ1
        # msg.velocity[21] = 0.0  # UPJ2
        # msg.velocity[22] = 0.0  # UPJ3
        # msg.velocity[23] = 0.0  # UPJ4
        # msg.velocity[24] = 0.0  # TOOLJ1

        self.publisher_.publish(msg)
        self.t += self.dt

    # upper_node.py 테스트용
    # def __init__(self):
    #     super().__init__('sin_velocity_publisher')

    #     self.publisher_ = self.create_publisher(JointState, '/hanyang/velocity_cmd', 10)

    #     self.joint_names = [
    #         # "FRJ1", "FRJ2", "FRJ3", "FRJ5", "FRJW",
    #         # "FLJ1", "FLJ2", "FLJ3", "FLJ5", "FLJW",
    #         # "RRJ1", "RRJ2", "RRJ3", "RRJ5", "RRJW",
    #         # "RLJ1", "RLJ2", "RLJ3", "RLJ5", "RLJW",
    #         "UPJ1", "UPJ2", "UPJ3", "UPJ4", "TOOLJ1"
    #     ]
    #     self.num_joints = len(self.joint_names) + 1

    #     self.freq = 0.25  # Hz
    #     self.amp = 1.0   # rad/s
    #     self.t = 0.0
    #     self.dt = 0.01
    #     self.create_timer(self.dt, self.publish_velocity)

    # def publish_velocity(self):
    #     msg = JointState()
    #     msg.name = self.joint_names
    #     vel_value = self.amp * math.sin(2 * math.pi * self.freq * self.t)
    #     msg.velocity = [vel_value] * self.num_joints
    #     # msg.velocity[0] = 0.0  # UPJ1
    #     msg.velocity[1] = 0.0  # UPJ2
    #     msg.velocity[2] = 0.0  # UPJ3
    #     msg.velocity[3] = 0.0  # UPJ4
    #     msg.velocity[4] = 0.0  # TOOLJ1
        
    #     self.publisher_.publish(msg)
    #     self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    node = SinVelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
