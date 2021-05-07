import os
from typing import Dict

from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}  # type: Dict[str, str]
with open(os.path.join('molgym', 'version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='molgym',
    version=version_dict['__version__'],
    description='',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.6'],
    author='Gregor Simm and Robert Pinsler',
    author_email='gncs2@cam.ac.uk, rp586@cam.ac.uk',
    python_requires='>=3.7',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gym',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'ase',
        'schnetpack',
    ],
    zip_safe=False,
    test_suite='pytest',
    tests_require=['pytest'],
)
