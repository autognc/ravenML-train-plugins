from setuptools import setup

# def dependencies(file):
#     with open(file) as f:
#         return f.read().splitlines()

# figured out how to add object-detection via:
# https://stackoverflow.com/questions/12518499/pip-ignores-dependency-links-in-setup-py

setup(
    name='rmltfbbox',
    version='0.3',
    description='Training plugin for ravenml',
    packages=['rmltfbbox'],
    install_requires=[
        'numpy==1.16.4',
        'cython==0.29.13',
        'tensorflow==1.14.0',
        'object-detection @ https://github.com/autognc/object-detection/tarball/object-detection#egg=object-detection',
        'absl-py==0.8.0',
        'pycocotools==2.0.0',
        'matplotlib==3.1.1',
        'contextlib2==0.5.5',
        'pillow==6.1.0',
        'lxml==4.4.0',
        'jupyter==1.0.0',
        'comet-ml==2.0.13',
        'opencv-python==4.1.2.30'        
    ],
    # dependency_links=[
    #     'https://github.com/autognc/object-detection/tarball/object-detection'
    # ],
    entry_points='''
        [ravenml.plugins.train]
        tf_bbox=rmltfbbox.core:tf_bbox
    '''
)
