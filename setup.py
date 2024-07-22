# setup.py
from setuptools import setup, find_packages

setup(
    name='Transformers_for_peptide_Classification',
    packages=['Transformers_for_peptide_Classification'],
    version='0.1.0',
    description='A package for peptide classification using transformers',
    author='Sandro Kuppel',
    install_requires=['torch',
                      'numpy',]
)