from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="bib",
    version="1.0.0",
    description="BIB: Biologically Inspired Brain — a neuroscience-grounded artificial brain in Python",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="UnikAI",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "numpy>=1.24",
        "numba>=0.57",
        "rich>=13.0",
    ],
    python_requires=">=3.10",
)
