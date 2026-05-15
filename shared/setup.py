import os
from glob import glob

from setuptools import setup


package_name = 'shared'

# Nested layout (D17, supersedes D1): Python sources live under
# shared/shared/ — matches `ros2 pkg create --build-type ament_python`
# convention and lets `colcon build --symlink-install` work (the flat
# package_dir={'shared': '.'} variant tripped colcon-core's
# _symlinks_in_build via os.path.join(build_base, '.') → EINVAL).
#
#   shared/
#   ├── shared/                   ← Python package root
#   │   ├── __init__.py
#   │   ├── robot_configs/
#   │   │   ├── schema.py         ← VLA-semantic yaml schema parser
#   │   │   ├── *_config.yaml     ← per-robot config
#   │   │   ├── urdf/             ← URDF XML per robot type
#   │   │   └── ffw_description/  ← mesh tree referenced by URDFs
#   │   ├── io/
#   │   │   └── file_io.py
#   │   └── logger/
#   ├── package.xml
#   ├── resource/                 ← ament index marker (repo-level)
#   └── setup.py

packages = [
    package_name,
    f'{package_name}.io',
    f'{package_name}.robot_configs',
]


def _walk_share_dir(src_subdir):
    """Yield (dest, [files]) tuples to install all files under src_subdir
    recursively into share/<package>/<rel>/<...>, where <rel> strips the
    leading <package_name>/ prefix from src so the share tree mirrors
    the importable layout (share/shared/robot_configs/...) rather than
    duplicating the package name (share/shared/shared/robot_configs/...).
    Used for the URDF mesh tree which is several levels deep."""
    pairs = []
    for root, _dirs, files in os.walk(src_subdir):
        if not files:
            continue
        rel_files = [os.path.join(root, f) for f in files]
        rel_root = os.path.relpath(root, package_name)
        pairs.append((f'share/{package_name}/{rel_root}', rel_files))
    return pairs


# shared/robot_configs/*_config.yaml      (per-robot ROS2 params loaded by
#                                          orchestrator launch + read directly
#                                          by policy main_runtime containers)
# shared/robot_configs/urdf/*.urdf        (per-robot URDF XML)
# shared/robot_configs/ffw_description/** (mesh tree, recursive)
# These live in shared/ rather than orchestrator/config/ or
# rosbag_recorder/config/ because they describe the robot, not any
# single consumer package.
robot_assets = [
    (
        f'share/{package_name}/robot_configs',
        glob(f'{package_name}/robot_configs/*_config.yaml'),
    ),
    (
        f'share/{package_name}/robot_configs/urdf',
        glob(f'{package_name}/robot_configs/urdf/*.urdf'),
    ),
] + _walk_share_dir(f'{package_name}/robot_configs/ffw_description')


setup(
    name=package_name,
    version='0.1.0',
    packages=packages,
    # Nested layout convention (D17): root namespace '' maps to current
    # directory, so 'shared' resolves to ./shared/ and subpackages
    # inherit. Avoids colcon-core's _symlinks_in_build veto on
    # `{'shared': 'shared'}` (key == value -> self-symlink).
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + robot_assets,
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
    description='Cyclo Intelligence shared utilities.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
