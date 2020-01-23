#!/usr/bin/env python
from setuptools import setup, find_packages
import warnings
from setuptools.command.install import install 

class PostInstallCommand(install):
    def run(self):
        try:
            from pip._internal import main as pipmain
            pipmain(['install', 'readline'])
        except:
            warnings.warn('Could not install readline')
        install.run(self)

setup(name='swutil',
      version='1.4.4',
      python_requires='>=3.6',
      description='Provide common code for validation, multiprocessing, logging, decorator, configuration, plotting, etc.',
      long_description=open('README.rst').read(),
      author='Soeren Wolfers',
      author_email='soeren.wolfers@gmail.com',
      url = 'https://github.com/soerenwolfers/swutil',
      packages=find_packages(exclude=['*tests']),#,'examples*']),
      cmdclass={ 'install': PostInstallCommand, },
      install_requires=['numpy','matplotlib','tikzplotlib','memory_profiler','pathos','dill','PyPDF2']
)
