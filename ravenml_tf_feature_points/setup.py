from setuptools import setup

setup(
    name='ravenML_tf_feature_points',
    version='0.1',
    description='Training plugin for ravenml',
    packages=['ravenml_tf_feature_points'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_feature_points=ravenml_tf_feature_points.core:tf_feature_points
    '''
)
