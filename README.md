# About
This repository contains Jupyter notebooks used to teach basic machine learning
techniques with TensorFlow, as well as support functions via the `TutorialSupport`
Python package. The use of the `TutorialSupport` Python package is designed to help
students focus only on the TensorFlow content rather than any dependencies, data
sources or plotting libraries. It also allows an interested instructor to install
all the required packages in one go.

Further, `TutorialSupport` slightly modifies the `tf.Session()` and
`tf.InteractiveSession()` classes to restrict CPU usage to only two CPUs by default,
which allows the training to occur on shared infrastructure (e.g. JupyterHub) in a
classroom whilst mitigating the potential for performance degradation incurred with
students simultaneously training their models. It also resets the numpy and TensorFlow
random seeds to help keep the results as reproducible as possible. The reason for
modifying these classes over providing an appropriate TensorFlow configuration object
is simply to ensure no extraneous setup tasks detract from the core objectives of the
coursework.

# Installing
The `TutorialSupport` package can be installed via pip using the following command
```
pip install -U git+https://github.com/MonashDataFluency/intro-to-tensorflow.git
```

GraphViz is also required and can be installed on Ubuntu systems using:
```
apt-get install graphviz
```

# The notebooks
Two notebooks are provided in this repository:
* `ML-Coursework_Complete.ipynb` - Contains full code that can be used as a reference
by the instructor
* `ML-Coursework_Student.ipynb` - Identical to the complete notebook except a number of
cells contain code that needs to be completed by the student

When maintaining this repository, it is essential that `ML-Coursework_Complete.ipynb`
and `ML-Coursework_Student.ipynb` are kept in sync.

## Content
The concepts covered in these notebooks are:

1. Sanity check [Essential] -
Run some simple "Hello World" code (1 minute)
2. Everything in TensorFlow is part of a graph [Essential]  -
Set up some basic calculations to get a feel for what a graph is (20 minutes)
3. A simple logistic regression model [Essential] -
Learn one of the fundamental machine learning models for classification and understand how TensorFlow optimises model parameters (30 minutes)
4. Extending to muliple classes [Essential] -
Adapt the simple logistic regression model to predict multiple classes and produce a reasonable handwriting recognition tool (30 minutes)
5. Generalising to a Multilayer Perceptron (MLP) model [Essential] -
Improve the handwriting recognition tool using a multilayered neural network (30 minutes)
6. Neural networks for image data [Recommended] -
Understand the basic components of convolutional neural networks and how they're used on imaging data; implement the LeNet architecture for handwriting recognition (45 minutes)
7. Introducing higher-level TensorFlow APIs [Recommended] -
Learn how to apply higher-level APIs to handle data pipelines, training, evaluation and inference (45 minutes)
8. [Bonus material] Neural networks for sequence data: Ratings from IMDB reviews [Optional] -
A prepared example covering the recurrent neural networks and their application to a text analysis task
