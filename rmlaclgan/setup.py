from setuptools import setup, find_packages

setup(
    name='rmlaclgan',
    version='0.1',
    description='ACL-GAN training plugin',
    packages=find_packages(),
    install_requires=[ 
        'numpy==1.20.3',
        'Pillow==8.2.0',
        'protobuf==3.17.3',
        'PyYAML==5.4.1',
        'six==1.16.0',
        'tensorboardX==2.2',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'typing-extensions==3.10.0.0'
    ],
    entry_points='''
        [ravenml.plugins.train]
        aclgan=rmlaclgan.core:aclgan
    '''
)