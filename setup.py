"""Package installer."""
from setuptools import setup, find_packages

# Set Parameters
NAME = 'pySUMMA'
DESCRIPTION = (
    'Methods for aggregating predictions by binary classification methods.'
)
PYTHON_VERSION = '>=3.5'
LICENSE_NAME = 'Apache License 2.0'
# VERSION
VERSION = '0.3.1'
# LONG DESCRIPTION
LONG_DESCRIPTION = ''
with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()
# REQUIREMENTS, relaxing version for pkg_resource
REQUIREMENTS = ['numpy>=1.14.0', 'scipy>=1.0.0', 'matplotlib>=2.2.0']

setup(
    name=NAME,
    version=VERSION,
    python_requires=PYTHON_VERSION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Robert Vogel and Mehmet Eren Ahsen',
    license=LICENSE_NAME,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License'
    ]
)
