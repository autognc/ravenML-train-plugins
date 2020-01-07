from setuptools import setup

setup(
    name='ravenML_tf_pose_regression',
    version='0.1',
    description='Training plugin for ravenml',
    packages=['ravenml_tf_pose_regression'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_pose_regression=ravenml_tf_pose_regression.core:tf_pose_regression
    '''
)
