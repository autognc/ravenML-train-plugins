from setuptools import setup, find_packages
from pkgutil import iter_modules


setup(
    name="rmltraintfmobilepose",
    version="0.1.1",
    description="Tensorflow keypoints regression plugin for ravenml",
    packages=find_packages(),
    install_requires=[
        "numpy",  # ==1.18.4',
        "tensorflow==2.5.1",
        "opencv-python",  # ==4.2.0.34',
        "tqdm",  # ==4.36.1',
        "matplotlib",  # ==2.2.2',
        "comet-ml",  # ==3.1.11',
        "moderngl",
    ],
    entry_points="""
        [ravenml.plugins.train]
        tf_mobilepose=rmltraintfmobilepose.core:tf_mobilepose
    """,
)
