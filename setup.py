import os

from setuptools import setup

setup(
    name='tensorflow-tutorial-support',
    version='',
    packages=['TutorialSupport'],
    package_data={'TutorialSupport': [
        'data/*.ipynb',
        'data/*.dat',
        'data/*.tfrecords',
        'data/imdb_model/*'
    ]},
    url='',
    license='',
    author='Jason Rigby',
    author_email='Jason.Rigby@monash.edu',
    description='Support functions for a TensorFlow tutorial',
    install_requires=[
        'tensorflow==1.6.0',
        'keras==2.1.4',
        'numpy==1.14.0',
        'pandas==0.22.0',
        'graphviz==0.8.2',
        'matplotlib==2.1.2',
        'seaborn==0.8.1',
        'python-mnist==0.5',
        'scikit-image==0.13.1',
        'appdirs==1.4.3',
        'nltk==3.2.5',
        'beautifulsoup4==4.6.0'
    ]
)
