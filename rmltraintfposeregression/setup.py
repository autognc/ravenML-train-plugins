from setuptools import setup, find_packages

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
        'tensorflow==2.1.0'
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_pose_regression=rmltraintfposeregression.core:tf_pose_regression
    '''
)
