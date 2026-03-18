from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    upper_node = Node(
        package='ecvt2_sim',
        executable='test_upper_node',
        name='sim_ecvt_upper_node',
        output='screen'
    )
    
    planner_node = Node(
        package='ecvt2_controller',
        executable='vp_sto_global_planner_node',
        name='vp_sto_global_planner_node',
        output='screen',
        parameters=[{
            'joint_state_topic': '/joint_states',
            'actuated_joints': ['UPJ1','UPJ2','UPJ3','UPJ4','TOOLJ1'],
            'passive_joints': ['UPJ5','UPJ6'],
            'goal_actuated': [0.4, 0.5, -0.5, 0.6, 0.0],
            'velocity_limits': [0.1, 0.1, 0.1, 0.1, 0.1],
            'acceleration_limits': [0.2, 0.2, 0.2, 0.2, 0.2],
            'n_via': 4,
            'n_eval': 100,
            'population': 40,
            'max_iterations': 100
        }]
    )
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
        upper_node,
        # dynamics_node,
        planner_node
    ])