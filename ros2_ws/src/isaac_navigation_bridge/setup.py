from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "isaac_navigation_bridge"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{PACKAGE_NAME}"],
        ),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pct_scan maintainers",
    maintainer_email="yagami-light7@users.noreply.github.com",
    description="Isaac Sim 到 SCAN Planner 的 ROS 2 里程计与点云边界节点。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
    "console_scripts": [
        "isaac_navigation_bridge = "
        "isaac_navigation_bridge.bridge_node:main",
        "odometry_tf_broadcaster = "
        "isaac_navigation_bridge.odometry_tf_broadcaster:main",
    ],
},
)
