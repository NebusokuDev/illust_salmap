from pathlib import Path

from setuptools import setup, find_packages

with Path("./requirements.txt").open("r") as require:
    requires_packages = require.read().splitlines()

setup(
    name='illust_salmap',
    version='1.0.0',
    description='A short description of your project',
    long_description='A short description of your project',
    author='Sumx21t',  # 作成者名
    author_email='otoro180yen@gmail.com',
    url='https://github.com/NebusokuDev/illust_salmap.git',
    packages=find_packages(),
    install_requires=requires_packages,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
