import os
from setuptools import setup, find_packages

# determine GPU or CPU install via env variable
gpu = os.getenv('RML_GPU')
tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

setup(
    name='rmltraintfinstance',
    version='0.1',
    description='Tensorflow Instance Segmentation training plugin for ravenML',
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
        tensorflow_pkg
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_instance=rmltraintfinstance.core:tf_instance
    '''
)
