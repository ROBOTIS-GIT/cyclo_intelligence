from glob import glob

from setuptools import setup


package_name = 'orchestrator'
authors_info = [
    ('Dongyun Kim', 'kdy@robotis.com'),
    ('Seongwoo Kim', 'kimsw@robotis.com')
]
authors = ', '.join(author for author, _ in authors_info)
author_emails = ', '.join(email for _, email in authors_info)

# Nested layout (D17, supersedes D2): Python sources live under
# orchestrator/orchestrator/ — matches `ros2 pkg create --build-type
# ament_python` convention and lets `colcon build --symlink-install`
# work (the flat package_dir={'orchestrator': '.'} variant tripped
# colcon-core's _symlinks_in_build via os.path.join(build_base, '.')
# → EINVAL).
#
# Layout:
#   orchestrator/
#   ├── orchestrator/        ← Python package (D17)
#   │   ├── __init__.py
#   │   ├── orchestrator_node.py
#   │   ├── bt/              (incl. bt/trees/, bt/bringup/ — data co-located with code)
#   │   ├── internal/
#   │   ├── timer/
#   │   └── training/
#   ├── launch/              ← repo-level (ament convention)
#   ├── config/, resource/
#   ├── scripts/             (standalone CLIs, no __init__.py)
#   ├── tests/, test/        (pytest)
#   ├── ui/                  (React app — no __init__.py)
#   ├── package.xml
#   └── setup.py

packages = [
    package_name,
    f'{package_name}.bt',
    f'{package_name}.bt.actions',
    f'{package_name}.bt.controls',
    f'{package_name}.internal',
    f'{package_name}.internal.communication',
    f'{package_name}.internal.device_manager',
    f'{package_name}.internal.file_browser',
    f'{package_name}.timer',
    f'{package_name}.training',
]

setup(
    name=package_name,
    version='0.1.0',
    packages=packages,
    # Nested layout convention (D17): root namespace '' maps to current
    # directory. Avoids colcon-core's _symlinks_in_build veto on
    # `{'orchestrator': 'orchestrator'}` (key == value -> self-symlink).
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch/ stays at repo level (ament convention) — glob unchanged.
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # Per-robot YAML params live in shared/robot_configs/ (alongside
        # urdf/, ffw_description/, and the schema.py helper). Launch
        # resolves them via get_package_share_directory('shared') +
        # 'robot_configs'.
        # BT assets absorbed from physical_ai_bt (Step 5-A). The trees /
        # bringup data dirs moved into nested orchestrator/orchestrator/bt/
        # (D17 user decision: data dirs that pair with code go nested).
        # Share install path stays the same so `get_package_share_directory(
        # 'orchestrator') / 'bt' / 'trees'` keeps working.
        (
            'share/' + package_name + '/bt/trees',
            glob(f'{package_name}/bt/trees/*.xml'),
        ),
        (
            'share/' + package_name + '/bt/bringup',
            glob(f'{package_name}/bt/bringup/*.yaml'),
        ),
    ],
    install_requires=[
        'setuptools',
        'interfaces',
        'mcap',
        'mcap-ros2-support',
        'matplotlib',
        'psutil',
    ],
    zip_safe=True,
    author=authors,
    author_email=author_emails,
    maintainer='Pyo',
    maintainer_email='pyo@robotis.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='ROS 2 package for Open Platform AI Kit integration',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orchestrator_node = orchestrator.orchestrator_node:main',
            'bt_node = orchestrator.bt.bt_node:main',
        ],
    },
)
