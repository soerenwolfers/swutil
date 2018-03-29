#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='swutil',
      version='1.1.3',
      description='Provide common code for validation, multiprocessing, logging, decorator, configuration, plotting, etc.',
      long_description=open('README.rst').read(),
      author='Soeren Wolfers',
      author_email='soeren.wolfers@gmail.com',
      url = 'https://bitbucket.org/soerenwolfers/swutil',
      packages=find_packages(exclude=['*tests']),#,'examples*']),
      install_requires=['numpy','matplotlib','matplotlib2tikz','memory_profiler']
)
