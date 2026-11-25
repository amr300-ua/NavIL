"""Ciclo 2: evaluación en bucle cerrado sobre Gazebo (Gazebo + AMCL ya lanzados).

    # Política (no hace falta Nav2):
    ros2 launch il_control closed_loop.launch.py scenario:=ruta_validada \\
        policy_type:=il_policy model_path:=$HOME/ros2_ws/models/mlp/best.pt repetitions:=5

    # Nav2 por las mismas rutas (requiere el remapeo de Nav2 a /cmd_vel_expert):
    ros2 launch il_control closed_loop.launch.py scenario:=ruta_validada controller:=expert

Lanza il_policy_node + cmd_vel_mux (arranca en 'stop') + episode_runner. Los
resultados quedan en output_dir/<run_tag>/{episodes.csv, summary.json}.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _setup(context):
    lc = lambda n: context.launch_configurations[n]  # noqa: E731
    share = get_package_share_directory('il_control')
    sim_time = lc('use_sim_time').lower() == 'true'
    policy_label = lc('policy_type') + (
        f":{os.path.basename(os.path.dirname(lc('model_path')))}/"
        f"{os.path.basename(lc('model_path'))}" if lc('model_path') else '')
    runner_params = {
        'use_sim_time': sim_time,
        'scenario': lc('scenario'),
        'controller': lc('controller'),
        'repetitions': int(lc('repetitions')),
        'output_dir': lc('output_dir'),
        'policy_label': policy_label if lc('controller') == 'policy' else 'nav2',
        'expert_uses_nav2': lc('expert_uses_nav2').lower() == 'true',
    }
    if lc('scenarios_file'):
        runner_params['scenarios_file'] = lc('scenarios_file')
    if lc('run_tag'):
        runner_params['run_tag'] = lc('run_tag')
    runner = Node(package='il_control', executable='episode_runner', name='episode_runner',
                  output='screen', emulate_tty=True, parameters=[runner_params])
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch',
                                                       'il_control.launch.py')),
            launch_arguments={'use_sim_time': lc('use_sim_time'), 'mode': 'stop',
                              'policy_type': lc('policy_type'),
                              'model_path': lc('model_path'),
                              'use_stamped': lc('use_stamped')}.items()),
        runner,
        # Al terminar el runner se cierra todo (útil para baterías automáticas)
        RegisterEventHandler(OnProcessExit(target_action=runner,
                                           on_exit=[EmitEvent(event=Shutdown(
                                               reason='episode_runner terminado'))])),
    ]


def generate_launch_description():
    args = [
        ('scenario', 'ruta_validada', 'Nombre en scenarios.yaml'),
        ('scenarios_file', '', 'Vacío = config/scenarios.yaml de il_control'),
        ('controller', 'policy', 'policy | expert'),
        ('repetitions', '1', 'Repeticiones de la ruta completa'),
        ('policy_type', 'il_policy', 'il_policy | mock_reactive | mock_goal | zero'),
        ('model_path', '', 'Checkpoint .pt de il_policy'),
        ('output_dir', '~/navil_results', ''),
        ('run_tag', '', 'Subcarpeta de resultados (vacío = automático)'),
        ('use_sim_time', 'true', ''),
        ('use_stamped', '', 'true=TwistStamped, false=Twist; vacío = YAML'),
        ('expert_uses_nav2', 'true', 'false = experto mock sin acción NavigateToPose'),
    ]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h)
                              for n, d, h in args] + [OpaqueFunction(function=_setup)])
