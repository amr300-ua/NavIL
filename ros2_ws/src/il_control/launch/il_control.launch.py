"""Lanza el módulo 4: nodo de inferencia + selector/seguridad.

    ros2 launch il_control il_control.launch.py                      # modo experto (Nav2)
    ros2 launch il_control il_control.launch.py mode:=policy \\
        policy_type:=il_policy model_path:=models/mlp/best.pt         # ciclo 2

Los argumentos vacíos no sobrescriben el YAML (``params_file``).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OVERRIDES = {
    # argumento: (nodo, parámetro)
    'mode': [('cmd_vel_mux', 'mode')],
    'policy_type': [('il_policy_node', 'policy_type')],
    'model_path': [('il_policy_node', 'model_path')],
}
STAMPED = [('il_policy_node', 'output_stamped'), ('cmd_vel_mux', 'expert_stamped'),
           ('cmd_vel_mux', 'policy_stamped'), ('cmd_vel_mux', 'output_stamped')]


def _as_bool(s: str) -> bool:
    if s.lower() in ('true', '1', 'yes'):
        return True
    if s.lower() in ('false', '0', 'no'):
        return False
    raise ValueError(f"Valor booleano no válido: '{s}'")


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    params = {'il_policy_node': {}, 'cmd_vel_mux': {}}
    for arg, targets in OVERRIDES.items():
        if lc(arg):
            for node, prm in targets:
                params[node][prm] = lc(arg)
    if lc('use_stamped'):
        for node, prm in STAMPED:
            params[node][prm] = _as_bool(lc('use_stamped'))
    sim_time = _as_bool(lc('use_sim_time'))
    params_file = lc('params_file')
    return [
        Node(package='il_control', executable='policy_node', name='il_policy_node',
             output='screen', emulate_tty=True,
             parameters=[params_file, params['il_policy_node'], {'use_sim_time': sim_time}]),
        Node(package='il_control', executable='cmd_vel_mux', name='cmd_vel_mux',
             output='screen', emulate_tty=True,
             parameters=[params_file, params['cmd_vel_mux'], {'use_sim_time': sim_time}]),
    ]


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('il_control'),
                                  'config', 'il_control.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true con Gazebo, false con fake_sim'),
        DeclareLaunchArgument('mode', default_value='', description='expert | policy | stop'),
        DeclareLaunchArgument('policy_type', default_value='',
                              description='mock_reactive | mock_goal | zero | il_policy'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('use_stamped', default_value='',
                              description='true=TwistStamped, false=Twist (todas las E/S)'),
        OpaqueFunction(function=_setup),
    ])
