
from setuptools import setup, find_packages

setup(
    name='scrapping_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'selenium',  # Add other dependencies here
    ],
    include_package_data=True,
)
