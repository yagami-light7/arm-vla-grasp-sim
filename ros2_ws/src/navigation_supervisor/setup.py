from glob import glob

from setuptools import setup


PACKAGE_NAME = "navigation_supervisor"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=[PACKAGE_NAME, "navigation_core"],
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
    zip_safe=False,
    maintainer="pct_scan maintainers",
    maintainer_email="yagami-light7@users.noreply.github.com",
    description="PCT、SCAN 与闭环控制器的 ROS 2 导航安全协调节点。",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigation_supervisor = navigation_supervisor.node:main",
        ],
    },
)
