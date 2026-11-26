from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'il_control'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=("test", "test.*")),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Alejandro Martinez',
    maintainer_email='amr300@alu.ua.es',
    description='Inferencia, selector experto/política y seguridad para TurtleBot 4',
    license='MIT',
    entry_points={
        'console_scripts': [
            'policy_node = il_control.policy_node:main',
            'cmd_vel_mux = il_control.cmd_vel_mux:main',
            'fake_sim = il_control.fake_sim:main',
            'episode_runner = il_control.episode_runner:main',
        ],
    },
)
