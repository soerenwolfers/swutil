#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='swutil',
      version='1.0.1',
      description='Provide common code logging, decorator, config, plotting, etc.',
      long_description=open('README.rst').read(),
      author='Soeren Wolfers',
      author_email='soeren.wolfers@gmail.com',
      packages=find_packages(exclude=['*tests']),#,'examples*']),
      install_requires=['numpy','matplotlib','matplotlib2tikz','memory_profiler']
)
