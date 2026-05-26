import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _compose_nodes(context, *args, **kwargs):
    urdf_pkg = LaunchConfiguration('urdf_package').perform(context) or 'forestry_robot_mjcf'
    urdf_relpath = LaunchConfiguration('urdf_relpath').perform(context) or 'xml/ecvt_v2.urdf'
    use_rviz = (LaunchConfiguration('rviz').perform(context) or 'true').lower() in ['true', '1', 'yes']

    urdf_pkg_share = get_package_share_directory(urdf_pkg)
    urdf_path = os.path.join(urdf_pkg_share, urdf_relpath)
    if not os.path.exists(urdf_path):
        raise RuntimeError(f"URDF not found: {urdf_path}")

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    this_pkg_share = get_package_share_directory('ecvt2_sim')
    rviz_config = os.path.join(this_pkg_share, 'rviz', 'ecvt.rviz')

    actions = [
        Node(
            package='ecvt2_sim',
            executable='test_robot_node',
            name='sim_ecvt_full_node',
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='ecvt2_controller',
            executable='base_arm_task_controller_full',
            name='base_arm_task_controller_full',
            output='screen',
            parameters=[{'urdf_path': urdf_path}],
        ),
    ]

    if use_rviz:
        rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=rviz_args,
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Run RViz2 if true.',
        ),
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
