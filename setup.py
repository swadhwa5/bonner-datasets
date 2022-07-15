from setuptools import setup, find_namespace_packages

requirements = (
    "requests",
    "boto3",
    "numpy",
    "pandas",
    "xarray",
    "scipy",
    "nipy",
    "h5py",
    # "brainio", ## TODO add github path
)

setup(
    name="bonner-datasets",
    version="0.1.0",
    packages=find_namespace_packages(),
    install_requires=requirements,
)
