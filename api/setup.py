from setuptools import setup, find_packages

setup(
    name='machine-learning-api',
    version='0.1',
    packages=find_packages(where="api/src/"),
)