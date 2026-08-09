from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "pct_ros2_adapter"


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
        (
            f"share/{PACKAGE_NAME}/upstream",
            [
                *glob("upstream/*.json"),
                *glob("upstream/*.md"),
            ],
        ),
        (
            f"share/{PACKAGE_NAME}/upstream/patches",
            glob("upstream/patches/*.patch"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="pct_scan maintainers",
    maintainer_email="yagami-light7@users.noreply.github.com",
    description="把 PCT 三维全局规划结果发布为 ROS 2 地面高度 Path。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pct_ros2_adapter = pct_ros2_adapter.node:main",
        ],
    },
)
