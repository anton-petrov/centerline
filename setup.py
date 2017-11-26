from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='centerline',
    version='0.2.1',
    description='Calculate the centerline of a polygon',
    long_description=long_description,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS'
    ],
    url='https://github.com/fitodic/centerline.git',
    author='Filip Todic',
    author_email='todic.filip@gmail.com',
    license='MIT',
    packages=['centerline'],
    install_requires=[
        'GDAL>=2.0.1',
        'Fiona>=1.7.0',
        'Shapely>=1.5.13',
        'numpy>=1.10.4',
        'scipy>=0.16.1',
    ],
    extras_require={
        'dev': [
            'autopep8',
            'flake8',
            'ipdb',
            'ipython[all]',
            'notebook',
            'jupyter',
            'isort'
        ],
        'test': [
            'coverage',
            'pytest>=3.0.0',
            'pytest-cov',
            'pytest-sugar',
            'pytest-runner'
        ],
    },
    scripts=[
        'bin/create_centerlines',
    ],
    include_package_data=True,
    zip_safe=False,
)
