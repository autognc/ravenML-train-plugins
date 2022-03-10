from setuptools import setup, find_packages
from pkgutil import iter_modules


setup(
    name="rmltrainpthrnet",
    version="0.1.1",
    description="pytorch keypoints regression plugin for ravenml",
    packages=find_packages(),
    install_requires=[
        "numpy",  # ==1.18.4',
        "torch",
        "opencv-python",  # ==4.2.0.34',
        "tqdm",  # ==4.36.1',
        "matplotlib",  # ==2.2.2',
        "comet-ml",  # ==3.1.11',
        'tfrecord',
        'pytorch-lightning',
        'torchvision',
        'attrdict'
    ],
    entry_points="""
        [ravenml.plugins.train]
        pt_hrnet=rmltrain.core:pt_hrnet
    """,
)
