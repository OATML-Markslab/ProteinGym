from distutils.core import setup


with open("README.md") as f:
    readme = f.read()

setup(
    name="kermut",
    version="0.1",
    packages=["kermut"],
    author="Peter Mørch Groth",
    author_email="petermoerchgroth@gmail.com",
    description="Codebase for Kermut",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/petergroth/kermut",
    python_requires=">=3.8",
)
