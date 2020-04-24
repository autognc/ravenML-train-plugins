import os
from setuptools import setup, find_packages

# determine GPU or CPU install via env variable
gpu = os.getenv('RMLTRAIN_BBOX_GPU')
tensorflow_pkg = 'tensorflow==2.0.0' if not gpu else 'tensorflow-gpu==2.0.0'

setup(
    name='rmltraintfposeregression',
    version='0.1',
    description='Tensorflow direct pose regression plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy==1.16.4',
        'pillow==6.0.0',
        'matplotlib==3.1.0',
        'tqdm==4.32.2',
        tensorflow_pkg
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_pose_regression=rmltraintfposeregression.core:tf_pose_regression
    '''
)
