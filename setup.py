from setuptools import setup, find_packages
from pathlib import Path

# Open the README to get short and long descriptions
README = Path('README.md')
if README.exists():
    with open(str(README), 'r') as f:
        long_description = f.read()
    short_description = [line for line in long_description.split('\n') if line][1]
else:
    long_description = short_description = ''

setup(
    name='covid19dataview',
    version='1.0.0',
    license='MIT',
    author='Ryan Lague',
    author_email='ryanlague@hotmail.com',
    short_description=short_description,
    long_description=long_description,
    packages=find_packages(),
    package_data={},
    url='https://github.com/ryanlague/covid19dataview',
    install_requires=[
    ]
)
