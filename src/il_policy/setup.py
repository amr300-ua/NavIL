from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'il_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='Elena Fernandez',
    maintainer_email='efc37@alu.ua.es',
    description='Política de Behavioral Cloning (MLP PyTorch) para TurtleBot 4',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'make_dummy_dataset = il_policy.synthetic:main',
            'train = il_policy.train:main',
            'evaluate = il_policy.evaluate:main',
            'predict_demo = il_policy.policy:main',
        ],
    },
)
