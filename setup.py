import os
from setuptools import setup, find_packages

# read version
version_file = os.path.join(os.path.dirname(__file__), "balm", "version.py")
with open(version_file) as f:
    exec(f.read())

# read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="balm-antibody",
    version=__version__,
    author="Bryan Briney",
    author_email="briney@scripps.edu",
    description="BALM: Baseline Antibody Language Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/briney/balm",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
)
