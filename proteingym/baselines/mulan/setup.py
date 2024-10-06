from pathlib import Path
import runpy

from setuptools import setup, find_packages


# automatically detect the lib name
dirs = {
    d.parent for d in Path(__file__).resolve().parent.glob('*/__init__.py')
    if d.parent.is_dir() and (d.parent / '__version__.py').exists()
}

assert len(dirs) == 1, dirs
folder, = dirs
name = folder.name
init = runpy.run_path(folder / '__version__.py')
if name == 'my-project-name':
    raise ValueError('Rename "my-project-name" to your project\'s name.')
if '__version__' not in init:
    raise ValueError('Provide a __version__ in __init__.py')

version = init['__version__']
with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=version,
    install_requires=requirements,
    # OPTIONAL: uncomment if needed
    python_requires='>=3.6',
)
