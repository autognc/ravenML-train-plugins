import os
from setuptools import setup, find_packages

# def dependencies(file):
#     with open(file) as f:
#         return f.read().splitlines()

# figured out how to add object-detection via:
# https://stackoverflow.com/questions/12518499/pip-ignores-dependency-links-in-setup-py

# figured out to use find_packages() via:
# https://stackoverflow.com/questions/10924885/is-it-possible-to-include-subdirectories-using-dist-utils-setup-py-as-part-of

# determine GPU or CPU install via env variable
gpu = os.getenv('RMLTRAIN_TF_BBOX_GPU')
tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

setup(
    name='rmltraintfbbox',
    version='0.3',
    description='Tensorflow Bounding Box training plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy==1.16.4',
        'cython==0.29.13',
        'object-detection @ https://github.com/autognc/object-detection/tarball/object-detection#egg=object-detection',
        'absl-py==0.8.0',
        'pycocotools-fix==2.0.0.1',
        'matplotlib==3.1.1',
        'contextlib2==0.5.5',
        'pillow==6.1.0',
        'lxml==4.4.0',
        'jupyter==1.0.0',
        'comet-ml==2.0.13',
        'opencv-python==4.1.2.30',
        tensorflow_pkg
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_bbox=rmltraintfbbox.core:tf_bbox
    '''
)
