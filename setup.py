from setuptools import find_packages
from setuptools import setup

setup(
    name="DADA",
    packages=find_packages(),
    url='https://github.com/valeoai/DADA',
    description='DADA: Depth-aware Domain Adaptation in Semantic Segmentation',
    long_description=open('README.md').read(),
    install_requires=[
        "pyyaml",
        "tensorboardX",
        "easydict",
        "matplotlib",
        "scipy",
        "scikit-image",
        "future",
        "setuptools",
        "tqdm",
        "cffi",
    ],
    include_package_data=True,
)
