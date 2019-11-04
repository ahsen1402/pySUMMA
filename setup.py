import os
import re
from setuptools import setup, find_packages
from pkg_resources import resource_string

# Set Parameters
NAME='pySUMMA'
DESCRIPTION = "Methods for aggregating predictions by binary classification methods."
PYTHON_VERSION="==3.7.3"
LICENSE_NAME="Apache License 2.0"

# VERSION
version_file = resource_string("pySUMMA", "__init__.py").decode("utf-8")
version_file = version_file.strip().split("\n")
j = 0
while "__version__" not in version_file[j]:
    j+=1
VERSION = re.search("[0-9.]+", version_file[j]).group()

# LONG DESCRIPTION
LONG_DESCRIPT = resource_string("pySUMMA", "../README.md").decode("utf-8").strip()

# REQUIREMENTS
REQUIREMENTS = resource_string("pySUMMA", "../requirements.txt").decode("utf-8")
REQUIREMENTS = REQUIREMENTS.strip().split("\n")

setup(name=NAME,
    version=VERSION,
    python_requires=PYTHON_VERSION,
    long_description=LONG_DESCRIPT,
    long_description_content_type='text/markdown',
    author='Robert Vogel and Mehmet Eren Ahsen',
    license=LICENSE_NAME,
    description = DESCRIPTION,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License"
    ]
)
