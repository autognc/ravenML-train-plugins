from setuptools import setup, find_packages

setup(
    name='rmltraintfkeypoints',
    version='0.1',
    description='Tensorflow keypoints regression plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy==1.16.4',
        'tensorflow==2.1.0'
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_pose_regression=rmltraintfkeypoints.core:tf_keypoints
    '''
)
