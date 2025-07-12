from setuptools import find_packages, setup

package_name = 'item_pose_service'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hma',
    maintainer_email='teramaru77@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_pose_server = item_pose_service.pose_server_node:main',
            'aruco_pose_client = item_pose_service.pose_client_node:main',
            'aruco_pose_client_wrs = item_pose_service.item_pose_client_node:main',
            'aruco_pose = item_pose_service.pose_node:main',
            'aruco_pose_server_wrs= item_pose_service.item_pose_server_node:main',
        ],
    },
)
