from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteingym",
    description="ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="OATML-Markslab",
    version="1.01",
    license="MIT",
    url="https://github.com/OATML-Markslab/ProteinGym",    
    packages=["proteingym"]
)