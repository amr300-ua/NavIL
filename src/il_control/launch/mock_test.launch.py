"""Prueba del ciclo 1 sin Gazebo ni Nav2: fake_sim + experto mock + il_control.

    ros2 launch il_control mock_test.launch.py                 # conduce el "experto"
    ros2 param set /cmd_vel_mux mode policy                    # pasa a la política
    ros2 launch il_control mock_test.launch.py mode:=policy policy_type:=mock_goal
        # política sin evasión: la capa de seguridad debe frenar antes del obstáculo
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('il_control')
    sim_params = os.path.join(share, 'config', 'mock_sim.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='expert'),
        DeclareLaunchArgument('policy_type', default_value='mock_reactive'),
        DeclareLaunchArgument('model_path', default_value=''),
        Node(package='il_control', executable='fake_sim', name='fake_sim',
             output='screen', emulate_tty=True, parameters=[sim_params]),
        Node(package='il_control', executable='policy_node', name='fake_expert',
             output='screen', emulate_tty=True, parameters=[sim_params]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'il_control.launch.py')),
            launch_arguments={
                'use_sim_time': 'false',
                'mode': LaunchConfiguration('mode'),
                'policy_type': LaunchConfiguration('policy_type'),
                'model_path': LaunchConfiguration('model_path'),
            }.items()),
    ])
