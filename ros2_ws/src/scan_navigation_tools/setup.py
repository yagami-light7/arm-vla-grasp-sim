from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "scan_navigation_tools"


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
    description="为 SCAN Planner 发布手工三维 nav_msgs/Path。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manual_path_publisher = "
            "scan_navigation_tools.manual_path_publisher:main",
        ],
    },
)
