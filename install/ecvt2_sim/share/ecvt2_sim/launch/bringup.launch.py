# sim_ecvt_controller/launch/bringup.launch.py
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def _compose_nodes(context, *args, **kwargs):
    # --- URDF 위치(기본: forestry_robot_mjcf/xml/ecvt_v2.urdf) ---
    urdf_pkg = LaunchConfiguration('urdf_package').perform(context) or 'forestry_robot_mjcf'
    urdf_relpath = LaunchConfiguration('urdf_relpath').perform(context) or 'xml/ecvt_v2.urdf'

    try:
        urdf_pkg_share = get_package_share_directory(urdf_pkg)
    except Exception as e:
        raise RuntimeError(f"[bringup] package '{urdf_pkg}'를 찾을 수 없습니다: {e}")

    urdf_path = os.path.join(urdf_pkg_share, urdf_relpath)
    if not os.path.exists(urdf_path):
        raise RuntimeError(f"[bringup] URDF가 없습니다: {urdf_path}")

    # --- RViz 설정 기본값: sim_ecvt_controller/rviz/ecvt.rviz ---
    this_pkg_share = get_package_share_directory('ecvt2_sim')
    default_rviz = os.path.join(this_pkg_share, 'rviz', 'ecvt.rviz')

    rviz_config_arg = LaunchConfiguration('rviz_config').perform(context)
    rviz_config = rviz_config_arg if rviz_config_arg else (default_rviz if os.path.exists(default_rviz) else '')

    # RViz 실행 여부
    use_rviz = (LaunchConfiguration('rviz').perform(context) or 'true').lower() in ['true', '1', 'yes']

    # --- robot_description 읽기 ---
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # ================== Nodes ==================
    # 1) 시뮬/컨트롤러 노드
    sim_node = Node(
        package='ecvt2_sim',
        executable='robot_node',
        name='sim_ecvt_node',
        output='screen'
    )

    # 2) robot_state_publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # 3) RViz2 (옵션)
    actions = [sim_node, rsp_node]
    if use_rviz:
        rviz_args = ['-d', rviz_config] if rviz_config else []
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=rviz_args
        ))

    return actions

def generate_launch_description():
    return LaunchDescription([
        # RViz 실행 여부
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Run RViz2 if true.'
        ),
        # RViz 설정 파일 경로 (미지정 시 sim_ecvt_controller/rviz/ecvt.rviz 사용)
        DeclareLaunchArgument(
            'rviz_config', default_value='',
            description='Path to RViz config file (.rviz/.rviz2).'
        ),
        # URDF를 제공하는 패키지(기본: forestry_robot_mjcf)
        DeclareLaunchArgument(
            'urdf_package', default_value='forestry_robot_mjcf',
            description='Package that contains the URDF and meshes.'
        ),
        # 패키지 내 URDF 상대 경로
        DeclareLaunchArgument(
            'urdf_relpath', default_value='xml/ecvt_v2.urdf',
            description='Relative path to URDF inside urdf_package share dir.'
        ),
        OpaqueFunction(function=_compose_nodes),
    ])
