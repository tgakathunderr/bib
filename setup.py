from setuptools import setup, find_packages

setup(
    name="bib",
    version="1.0.0",
    description="BIB: Biologically Inspired Brain — 1:1 neuroscience-grounded organism brain",
    author="UnikAI",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "numba>=0.57",
        "rich>=13.0",
    ],
    python_requires=">=3.9",
)
