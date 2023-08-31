from setuptools import find_packages, setup

setup(
    name="facsimile",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "scikit-learn", "matplotlib", "tqdm"],
)
