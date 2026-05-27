import os

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    hsl_lib_dir = '/home/jin/harco/coinhsl/install/lib/x86_64-linux-gnu'
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    hsl_library_path = (
        hsl_lib_dir if not ld_library_path else f'{hsl_lib_dir}:{ld_library_path}'
    )

    upper_node = Node(
        package='ecvt2_sim',
        executable='test_upper_node',
        name='sim_ecvt_upper_node',
        output='screen',
        parameters=[{
            'passive_csv_enabled': True,
            'passive_csv_path': '/home/jin/harco/ecvt2_ws/upj5_upj6_joint_states.csv'
        }]
    )
    
    # planner_node = Node(
    #     package='ecvt2_controller',
    #     executable='vp_sto_global_planner_node',
    #     name='vp_sto_global_planner_node',
    #     output='screen',
    #     parameters=[{
    #         'joint_state_topic': '/joint_states',
    #         'actuated_joints': ['UPJ1','UPJ2','UPJ3','UPJ4','TOOLJ1'],
    #         'passive_joints': ['UPJ5','UPJ6'],
    #         'goal_actuated': [0.4, 0.5, -0.5, 0.6, 0.0],
    #         'velocity_limits': [5.0, 5.0, 5.0, 5.0, 5.0],
    #         'acceleration_limits': [1.0, 1.0, 1.0, 1.0, 1.0],
    #         'n_via': 3,
    #         'n_eval': 61,
    #         'population': 60,
    #         'max_iterations': 1000,
    #         'parallel_workers': 20,
    #         'random_seed': -1,
    #         'sigma0': 0.1,
    #         'w_time': 0.5,
    #         'w_smooth': 0.01,
    #         # 'w_terminal': 1.0,
    #         # 'w_passive_track': 1.0,
    #         # 'w_passive_damping': 10.0,
    #         'w_post_terminal_track': 1.0,
    #         'w_post_terminal_energy': 10.0,
    #         'w_via_regularization': 1e-4,
    #         'post_terminal_duration': 2.0,
    #         'post_terminal_steps': 50
    #     }]
    # )

    paper_planner_node = Node(
        package='ecvt2_controller',
        executable='vp_sto_paper_global_planner_node',
        name='vp_sto_paper_global_planner_node',
        output='screen',
        remappings=[
            ('~/actuated_reference', '/vp_sto_global_planner_node/actuated_reference'),
            ('~/planner_debug', '/vp_sto_global_planner_node/planner_debug')
        ],
        parameters=[{
            'joint_state_topic': '/joint_states',
            'actuated_joints': ['UPJ1','UPJ2','UPJ3','UPJ4','TOOLJ1'],
            'goal_actuated': [0.4, 0.5, -0.5, 0.6, 0.0],
            'velocity_limits': [5.0, 5.0, 5.0, 5.0, 5.0],
            'acceleration_limits': [1.0, 1.0, 1.0, 1.0, 1.0],
            'n_via': 3,
            'n_eval': 61,
            'population': 60,
            'max_iterations': 1000,
            'ipopt_linear_solver': 'ma97',
            'random_seed': -1,
            'sigma0': 0.1,
            'stop_sigma': 1e-3,
            'joint_limit_penalty_gain': 1e6
        }]
    )
    
    # 3457753494
    # 608337449
    # 2641194423
    # dynamics_node = Node(
    #     package='ecvt2_controller',
    #     executable='live_upper_dynamics_node',
    #     name='live_upper_dynamics_node',
    #     output='screen',
    #     parameters=[{
    #         'urdf_path': '/home/jin/harco/ecvt2_ws/forestry_robot_mjcf/xml/ecvt_v2_upper.urdf',
    #         'joint_state_topic': '/joint_states',
    #         'state_joints': ['UPJ1','UPJ2','UPJ3','UPJ4','UPJ5','UPJ6','TOOLJ1'],
    #         'actuated_joints': ['UPJ1','UPJ2','UPJ3','UPJ4'],
    #         'passive_joints': ['UPJ5','UPJ6'],
    #         'print_full_matrix': False,
    #         'log_every_n': 20,
    #     }]
    # )

    return LaunchDescription([
        SetEnvironmentVariable('LD_LIBRARY_PATH', hsl_library_path),
        upper_node,
        # dynamics_node,
        paper_planner_node,
        # planner_node
    ])
