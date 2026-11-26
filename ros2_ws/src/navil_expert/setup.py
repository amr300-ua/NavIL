from setuptools import find_packages, setup

package_name = 'navil_expert'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            ['config/scenarios.yaml']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lucasbeneyto',
    maintainer_email='lbs62@alu.ua.es',
    description='Controlador experto basado en Nav2 para generar demostraciones '
                'de navegacion autonoma mediante aprendizaje por imitacion.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'goal_sender = navil_expert.goal_sender:main',
            'trajectory_generator = navil_expert.trajectory_generator:main',
        ],
    },
)
