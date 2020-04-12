from setuptools import setup

# def dependencies(file):
#     with open(file) as f:
#         return f.read().splitlines()

setup(
    name='rmltfbbox',
    version='0.3',
    description='Training plugin for ravenml',
    packages=['rmltfbbox'],
    entry_points='''
        [ravenml.plugins.train]
        tf_bbox=rmltfbbox.core:tf_bbox
    '''
)
