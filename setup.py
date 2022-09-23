import os
import io
import sys
from setuptools import find_packages, setup, Command

with open('requirements.txt') as rf:
      requirements = rf.read().splitlines()

with open('README.md') as rf:
    readme = rf.read()

with open('LICENSE') as rf:
    license = rf.read()


setup(name='cosde',
      version='0.1',
      description='conditional orthogonal series density estimator',
      author='Katherine Tsai',
      author_email='kt14@illinois.edu',
      long_description=readme,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python3.7',
          'License :: OSI Approved :: MIT License',
          'Operating System :: Unix',
          'Operating System :: iOS'
          ],
    packages=['cosde'],
    license=license,
    install_requires=requirements)

