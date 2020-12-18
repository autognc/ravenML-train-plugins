from setuptools import setup, find_packages


setup(
    name='rmltraintfkeypoints',
    version='0.1.1',
    description='Tensorflow keypoints regression plugin for ravenml',
    packages=find_packages(),
    install_requires=[
        'numpy==1.18.4',
        'tensorflow==2.4.0',
        'opencv-python==4.2.0.34',
        'tqdm==4.36.1',
        'matplotlib==2.2.2',
        'comet-ml==3.1.11',
        'keras-unet==0.1.1'
    ],
    entry_points='''
        [ravenml.plugins.train]
        tf_keypoints=rmltraintfkeypoints.core:tf_keypoints
    '''
)
