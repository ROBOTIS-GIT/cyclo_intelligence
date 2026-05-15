from glob import glob

from setuptools import setup


package_name = 'cyclo_data'

# Nested layout (D17, supersedes D1): Python sources live under
# cyclo_data/cyclo_data/ — matches `ros2 pkg create --build-type
# ament_python` convention and lets `colcon build --symlink-install`
# work (the flat package_dir={'cyclo_data': '.'} variant tripped
# colcon-core's _symlinks_in_build via os.path.join(build_base, '.')
# → EINVAL).
#
# The C++ subpackage at ./recorder/rosbag_recorder/ stays at the
# repo top level — it has its own package.xml + CMakeLists.txt and is
# colcon-built via an explicit --paths entry. The naming overlap with
# the Python cyclo_data.recorder.* subpackage (which now lives at
# cyclo_data/cyclo_data/recorder/) is a layer distinction, not a
# conflict: top-level recorder/ is just the CMake-package container.

packages = [
    package_name,
    f'{package_name}.converter',
    f'{package_name}.converter.scripts',
    f'{package_name}.converter.video_encoder',
    f'{package_name}.editor',
    f'{package_name}.editor.scripts',
    f'{package_name}.hub',
    f'{package_name}.reader',
    f'{package_name}.recorder',
    f'{package_name}.services',
    f'{package_name}.visualization',
    f'{package_name}.visualization.scripts',
]

setup(
    name=package_name,
    version='0.1.1',
    packages=packages,
    # Nested layout convention (D17): root namespace '' maps to current
    # directory. Avoids colcon-core's _symlinks_in_build veto on
    # `{'cyclo_data': 'cyclo_data'}` (key == value -> self-symlink).
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch/ stays at repo level (ament convention) — glob unchanged.
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # hub/templates/ moved into nested package alongside hub/*.py
        # (D17 user decision: data dirs that pair with code go nested).
        (
            'share/' + package_name + '/hub/templates',
            glob(f'{package_name}/hub/templates/*.md'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Dongyun Kim',
    author_email='kdy@robotis.com',
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Cyclo data processing / recording / conversion / hub ROS 2 node.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cyclo_data_node = cyclo_data.cyclo_data_node:main',
            # CLI tools relocated from orchestrator/scripts/ in Step 7.
            # D9 (§10.3): 'visualize_rosbag' was on orchestrator/setup.py
            # before Step 3 Part A and lost its entry — re-registered here
            # alongside the other data-side utilities.
            'visualize_rosbag = cyclo_data.visualization.scripts.visualize_rosbag:main',
            'convert_rosbag_to_lerobot = cyclo_data.converter.scripts.convert_rosbag_to_lerobot:main',
            'remove_head_lift_joints = cyclo_data.editor.scripts.remove_head_lift_joints:main',
        ],
    },
)
