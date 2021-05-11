import setuptools
import pathlib


setuptools.setup(
    name='robodesk',
    version='0.0.1',
    description='Multi-task reinforcement learning benchmark.',
    url='https://github.com/google-research/robodesk',
    author='Harini Kannan',
    author_email='hkannan@google.com',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'dm_control', 'gym'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
