from setuptools import setup

# def dependencies(file):
#     with open(file) as f:
#         return f.read().splitlines()

setup(
    name='ravenML_tf_instance',
    version='0.1',
    description='Training plugin for ravenml',
    packages=['ravenml_tf_instance'],
    entry_points='''
        [ravenml.plugins.train]
        tf_instance=ravenml_tf_instance.core:tf_instance
    '''
)
