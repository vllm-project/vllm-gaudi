import importlib.util
import logging
import os
import subprocess
import sys
from sysconfig import get_paths
from typing import Dict, List

from setuptools import setup, find_packages
from setuptools_scm import get_version

try:
    VERSION = get_version(write_to="vllm_hpu/_version.py")
except LookupError:
    # The checkout action in github action CI does not checkout the tag. It
    # only checks out the commit. In this case, we set a dummy version.
    VERSION = "0.0.0"

ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
ext_modules = []

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_hpu.")
    return requirements
  
setup(
    name="vllm_hpu",
    version=VERSION,
    author="Intel",
    long_description="HPU plugin package for vLLM.",
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm-hpu",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm-hpu",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    extras_require={},
    entry_points={
        "vllm.platform_plugins": ["hpu = vllm_hpu:register"],
        "vllm.general_plugins": ["hpu_custom_ops = vllm_hpu:register_ops"],
    },
)