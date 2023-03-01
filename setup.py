#!/usr/bin/env python
"""Setup config file."""

from os import path

from setuptools import find_packages, setup


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='reachy-face-tracking',
    version='0.3.1',
    packages=find_packages(exclude=['tests']),

    install_requires=[
        'numpy',
    ],
)
