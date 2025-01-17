from setuptools import setup, find_packages

setup(
    name="scrapping-tools",  # Name of the package
    version="0.1.0",
    author="Juan Guillermo",
    author_email="juanosio838@gmail.com",
    description="Web scraping tools using Selenium.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/juanguillermo3/lab_market_trends/tree/main/scrapping_tools",
    packages=find_packages(),  # This ensures all Python files inside are included
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "selenium>=4.0.0",  # Your package dependencies
    ],
)
