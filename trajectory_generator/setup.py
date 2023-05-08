from setuptools import setup, find_packages

setup(
    name='trajectory_generator',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'simplekml',
        'matplotlib'
    ],
)
