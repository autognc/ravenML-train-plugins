import os
from setuptools import setup, find_packages

# determine GPU or CPU install via env variable
gpu = os.getenv('RMLTRAIN_TF_SEMANTIC_GPU')
tensorflow_pkg = 'tensorflow==1.14.0' if not gpu else 'tensorflow-gpu==1.14.0'

setup(
    name='rmltraintfsemantic',
    version='0.1',
    description='Tensorflow Semantic Segmentation training plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy==1.16.4',
        'deeplab @ https://github.com/autognc/models/archive/deeplab-0.0.1.tar.gz#subdirectory=research/deeplab',
        'pillow==6.0.0',
        'matplotlib==3.1.0',
        'tqdm==4.32.2',
        tensorflow_pkg
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_semantic=rmltraintfsemantic.core:tf_semantic
    '''
)
