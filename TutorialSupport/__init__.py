import tempfile
import random
import shutil
import os

import tensorflow as tf
import keras
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.io import imread

from TutorialSupport.visualisation import plot_graph, plot_loss_accuracy, plot_misclassifications
from TutorialSupport.utils import load_mnist, do_convolution, load_imdb_data,\
    load_wine_data, count_parameters, tokenise, encode_text, get_data_location, get_imdb_pretrained, download_url


def monkey_patch_session(old_sess):
    _old_init = old_sess.__init__

    def _new_tf_session_init(self, target='', graph=None, config=None):
        if config is None:
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads = 2

        g = graph or tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        _old_init(self, target, graph, config)
    return _new_tf_session_init


tf.InteractiveSession.__init__ = monkey_patch_session(tf.InteractiveSession)
tf.Session.__init__ = monkey_patch_session(tf.Session)
