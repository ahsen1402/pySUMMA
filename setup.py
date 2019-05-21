import os
from setuptools import setup, find_packages
import pkg_resources

# ========================================
# Set Parameters
# ========================================

NAME='pySUMMA'
DESCRIPTION = "Implementation of methods for the aggergation of predictions by binary classification methods."
PYTHON_VERSION=">=3.7.0"
LICENSE_NAME="Apache License 2.0"

# ========================================
# VERSION
# ========================================

if pkg_resources.resource_exists("pySUMMA", "__version__"):
    VERSION = pkg_resources.resource_string("pySUMMA", "__version__").decode("utf-8").strip()
else:
    raise FileNotFoundError("The file '__version__' could not be found within the pysumma directory")


# ========================================
# LONG DESCRIPTION
# ========================================

if pkg_resources.resource_exists("pySUMMA", "../README.md"):
    LONG_DESCRIPT = pkg_resources.resource_string("pySUMMA", "../README.md").decode("utf-8").strip()
else:
    raise FileNotFoundError("The file 'README.md' could not be found within the pysumma directory")


# ========================================
# REQUIREMENTS
# ========================================

if pkg_resources.resource_exists("pySUMMA", "../requirements.txt"):
    REQUIREMENTS = pkg_resources.resource_string("pySUMMA", "../requirements.txt").decode("utf-8").strip()
    REQUIREMENTS = REQUIREMENTS.split("\n")
else:
    raise FileNotFoundError("The file 'requirements.txt' could not be found within the pysumma directory")

# ========================================
# SETUP
# ========================================

setup(name=NAME,
    version=VERSION,
    python_requires=PYTHON_VERSION,
    long_description=LONG_DESCRIPT,
    long_description_content_type='text/markdown',
    author='Robert Vogel, Mehmet Eren Ahsen, and Gustavo Stolovitzky',
    license=LICENSE_NAME,
    description = DESCRIPTION,
    packages=find_packages(exclude=("examples")),
    package_data={
        '':['__version__']
    },
    install_requires=REQUIREMENTS,
#     project_urls={
#         "Source Code": ""
#     },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License"
    ]
)
