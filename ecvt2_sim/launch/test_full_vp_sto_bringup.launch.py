import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _compose_nodes(context, *args, **kwargs):
    urdf_pkg = LaunchConfiguration('urdf_package').perform(context) or 'forestry_robot_mjcf'
    urdf_relpath = LaunchConfiguration('urdf_relpath').perform(context) or 'xml/ecvt_v2.urdf'

    urdf_pkg_share = get_package_share_directory(urdf_pkg)
    urdf_path = os.path.join(urdf_pkg_share, urdf_relpath)
    if not os.path.exists(urdf_path):
        raise RuntimeError(f"URDF not found: {urdf_path}")

    return [
        Node(
            package='ecvt2_sim',
            executable='test_robot_node',
            name='sim_ecvt_full_node',
            output='screen',
            parameters=[{
                'trajectory_topic': '/vp_sto_global_planner_full_node/actuated_reference',
                'traj_tracking_enabled': True,
                'traj_override_velocity_cmd': True,
                'traj_position_kp': 1.5,
                'passive_csv_enabled': True,
                'passive_csv_path': '/home/jin/harco/ecvt2_ws/upj5_upj6_joint_states.csv',
            }],
        ),
        Node(
            package='ecvt2_controller',
            executable='vp_sto_global_planner_full_node',
            name='vp_sto_global_planner_full_node',
            output='screen',
            parameters=[{
                'urdf_path': urdf_path,
                'joint_state_topic': '/joint_states',
                'n_via': 3,
                'n_eval': 61,
                'max_iterations': 100,
                'ipopt_print_level': 5,
                'ipopt_tolerance': 1e-3,
                'ipopt_fd_step': 1e-4,
                'ipopt_via_bound_margin': 1.0,
                'ipopt_max_cpu_time': 1000.0,
                'freeze_static_joint_tolerance': 1e-5,
                'random_seed': -1,
                'sigma0': 0.1,
                'w_time': 0.5,
                'w_smooth': 0.01,
                'w_post_terminal_track': 1.0,
                'w_post_terminal_energy': 10.0,
                'w_via_regularization': 1e-4,
                'post_terminal_duration': 2.0,
                'post_terminal_steps': 50,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'urdf_package',
            default_value='forestry_robot_mjcf',
            description='Package that contains the full-body URDF.',
        ),
        DeclareLaunchArgument(
            'urdf_relpath',
            default_value='xml/ecvt_v2.urdf',
            description='Full-body URDF path relative to urdf_package share directory.',
        ),
        OpaqueFunction(function=_compose_nodes),
    ])
