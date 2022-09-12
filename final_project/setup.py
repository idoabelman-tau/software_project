from setuptools import setup, find_packages, Extension

setup(
    name='myspkmeans',
    version='3.1.4',
    author="Ido Abelman, Shachar Haim",
    author_email="author@example.com",
    description="Normalized spectral clustering implementation",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension(
            'myspkmeans',
            ['spkmeansmodule.c', 'spkmeans.c', 'matrix.c'],
        ),
    ]
)