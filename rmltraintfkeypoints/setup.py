from setuptools import setup, find_packages


setup(
    name='rmltraintfkeypoints',
    version='0.1.1',
    description='Tensorflow keypoints regression plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'comet-ml',
        'keras-unet',
        'keras_applications'
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_keypoints=rmltraintfkeypoints.core:tf_keypoints
    '''
)
