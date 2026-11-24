"""Prueba sin Gazebo ni Nav2: fake_sim + experto mock + il_control.

    ros2 launch il_control mock_test.launch.py                 # conduce el "experto"
    ros2 param set /cmd_vel_mux mode policy                    # pasa a la política
    ros2 launch il_control mock_test.launch.py mode:=policy policy_type:=mock_goal
        # política sin evasión: la capa de seguridad debe frenar antes del obstáculo

    # Ciclo 2: episodios automáticos con métricas (runner) sobre el stub
    ros2 launch il_control mock_test.launch.py runner:=true scenario:=mock_circuito
    ros2 launch il_control mock_test.launch.py runner:=true controller:=expert
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _setup(context):
    lc = lambda n: context.launch_configurations[n]  # noqa: E731
    share = get_package_share_directory('il_control')
    sim_params = os.path.join(share, 'config', 'mock_sim.yaml')
    runner = lc('runner').lower() == 'true'
    actions = [
        Node(package='il_control', executable='fake_sim', name='fake_sim',
             output='screen', emulate_tty=True,
             parameters=[sim_params, {'publish_goal': not runner}]),
        Node(package='il_control', executable='policy_node', name='fake_expert',
             output='screen', emulate_tty=True, parameters=[sim_params]),
    ]
    common = {'use_sim_time': 'false', 'policy_type': lc('policy_type'),
              'model_path': lc('model_path')}
    if runner:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch',
                                                       'closed_loop.launch.py')),
            launch_arguments={**common, 'scenario': lc('scenario'),
                              'controller': lc('controller'),
                              'repetitions': lc('repetitions'),
                              'expert_uses_nav2': 'false'}.items()))
    else:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch',
                                                       'il_control.launch.py')),
            launch_arguments={**common, 'mode': lc('mode')}.items()))
    return actions


def generate_launch_description():
    args = [('mode', 'expert'), ('policy_type', 'mock_reactive'), ('model_path', ''),
            ('runner', 'false'), ('scenario', 'mock_cruce'), ('controller', 'policy'),
            ('repetitions', '1')]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d) for n, d in args]
                             + [OpaqueFunction(function=_setup)])
