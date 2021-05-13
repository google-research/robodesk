import setuptools
import pathlib

import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('robodesk/assets')

print(extra_files)


setuptools.setup(
    name='robodesk',
    version='0.0.5',
    description='Multi-task reinforcement learning benchmark.',
    url='https://github.com/google-research/robodesk',
    author='Harini Kannan',
    author_email='hkannan@google.com',
    packages=['robodesk'],
    package_data={'robodesk': extra_files},
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
