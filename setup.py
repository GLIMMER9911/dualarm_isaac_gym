"""Installation script for the 'isaacgymenvs' python package."""

from setuptools import setup, find_packages
import os

root_dir = os.path.dirname(os.path.realpath(__file__))

INSTALL_REQUIRES = [
    # RL and Simulation
    "gym==0.23.1",
    "torch>=2.1.0",
    "omegaconf>=2.1.1",
    "termcolor",
    "hydra-core>=1.1",
    "rl-games==1.6.0",
    
    # Visualization & Analysis
    "matplotlib",
    "pandas",
    "dtw-python",

    # Utilities
    "pyvirtualdisplay",
    "icecream",
    "pickle5; python_version<'3.8'",  # Only needed for older Python
]

setup(
    name="isaacgymenvs",
    version="1.3.4",
    author="NVIDIA",
    description="Benchmark environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["robotics", "reinforcement learning", "isaacgym", "rl"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
    ],
    zip_safe=False,
)