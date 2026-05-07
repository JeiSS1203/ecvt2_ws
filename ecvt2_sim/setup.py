from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ecvt2_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='harco',
    maintainer_email='gunwoo00@hanyang.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_node = ecvt2_sim.robot_node:main',
            'upper_node = ecvt2_sim.upper_node:main',
            'test_upper_node = ecvt2_sim.test_upper_node:main',
            'test_robot_node = ecvt2_sim.test_robot_node:main',
            'sin_velocity_publisher = ecvt2_sim.sin_velocity_publisher:main',
            'plot_passive_joints = ecvt2_sim.plot_passive_joints:main',
            'plot_passive_csv = ecvt2_sim.plot_passive_csv:main',
        ],
    },
)
