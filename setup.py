from setuptools import setup

description = \
    """A codebase corresponding to Humphrey, E. J. "An Exploration of Deep
    Learning in Music Informatics", PhD Dissertation, NYU (2015)."""

setup(
    name='dl4mir',
    version='0.1.0',
    description=description,
    author='Eric J. Humphrey',
    author_email='ejhumphrey@nyu.edu',
    url='http://github.com/ejhumphrey/dl4mir',
    download_url='http://github.com/ejhumphrey/dl4mir/releases',
    packages=[
        'dl4mir',
        'dl4mir.chords',
        'dl4mir.common',
        'dl4mir.timbre',
        'dl4mir.guitar'
    ],
    package_data={
        'dl4mir.guitar': ['*.json']
    },
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7"
    ],
    keywords='machine learning, neural network',
    license='ISC',
    install_requires=[
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'scikit-learn >= 0.14.0',
        'matplotlib',
        'theano >= 0.6.0',
        'joblib',
        'tabulate'
    ]
)
