from graphviz import Digraph
import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_graph(graph=None):
    if graph is None:
        graph = tf.get_default_graph()
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot


def plot_loss_accuracy(train_loss, train_acc, test_loss, test_acc):
    plt.figure(1, figsize=(20, 10))
    plt.subplot(211)
    plt.plot(test_loss)
    plt.plot(train_loss)
    plt.legend(['test', 'train'], loc='upper right')
    plt.ylabel('loss')

    plt.subplot(212)
    plt.plot(test_acc)
    plt.plot(train_acc)
    plt.legend(['test', 'train'], loc='lower right')
    plt.ylabel('accuracy')

    return plt


def plot_misclassifications(true_labels, predicted_labels):
    crosstab = pd.crosstab(pd.Series(true_labels, name="Actual"),
                           pd.Series(predicted_labels, name="Predicted"))
    vmax = crosstab.as_matrix()[~np.eye(10, dtype=bool)].max()
    return sns.heatmap(crosstab, annot=True, square=True, fmt='1d', vmin=0, vmax=vmax)
