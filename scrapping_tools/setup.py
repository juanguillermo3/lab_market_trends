from setuptools import setup, find_packages

setup(
    name='scrapping_tools',
    version='0.1.0',
    author='Juan Guillermo',
    author_email='juanosio838@gmail.com',
    description='A library for Selenium-based web navigation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/juanguillermo3/lab_market_trends/new/main/scrapping_tools',
    packages=find_packages(),
    install_requires=[
        'selenium>=4.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
