import os
import tarfile
import urllib
import random
import shutil

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from appdirs import user_cache_dir
from bs4 import BeautifulSoup
from mnist import MNIST
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

IMDB_VOCAB = None
IMDB_ALL_FILES = None
IMDB_TRAIN_FILES = None
IMDB_TEST_FILES = None

sentence_tokeniser = nltk.tokenize.punkt.PunktSentenceTokenizer()
lemmatiser = nltk.stem.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english', ignore_stopwords=True)


def get_data_location(file_name=None, prefer_cache=True, raise_if_not_found=True, isdir=False):
    """
    Returns the path to a required file
    :param file_name: name of the file to search for
    :param prefer_cache: whether to look in the user's cache or in the package's default data directory first
    :param raise_if_not_found: raise an error if the file is not found
    :param isdir: are we looking for a file or directory?
    :return: a file path
    """
    exists = os.path.isdir if isdir else os.path.isfile
    local_data_location = os.path.join(os.path.dirname(__file__), 'data')
    cache_location = user_cache_dir("TensorFlowTutorial", "Monash")
    if not os.path.isdir(cache_location):
        os.makedirs(cache_location)
    if file_name is None:
        return cache_location if prefer_cache else local_data_location
    else:
        local_file = os.path.join(local_data_location, file_name)
        cache_file = os.path.join(cache_location, file_name)
        check_locations = [local_file, cache_file]
        if prefer_cache:
            check_locations = reversed(check_locations)
        for l in check_locations:
            if exists(l) or not raise_if_not_found:
                return l
        raise FileNotFoundError()


def download_url(url, target_file_name=None, target_dir=None):
    """
    Downloads or returns the path to a required file
    :param url: URL from which to download the file
    :param target_file_name: name of the target file (defaults to file name from the URL)
    :return: a file path
    """
    if target_file_name is None:
        target_file_name = url.split("/")[-1]
    target_path = target_dir or os.path.join(get_data_location(), target_file_name)
    if not os.path.isfile(target_path):
        print("Downloading %s from %s ... " % (target_file_name, url), end='')
        urllib.request.urlretrieve(url, target_path)
        print("Done.")
    return target_path


def load_wine_data(test_frac=None):
    """
    Downloads the wine dataset
    :param test_frac: fraction to reserve for testing
    :return: a pandas data frame if no test fraction is specified, otherwise a numpy array
    """
    red_wine = pd.read_csv(
        download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"),
        delimiter=";")
    red_wine['Type'] = 1
    white_wine = pd.read_csv(
        download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),
        delimiter=";")
    white_wine['Type'] = 0
    wine = pd.concat([red_wine, white_wine]).sample(frac=1, random_state=1234)
    if test_frac is None:
        return wine

    test_split = int(test_frac * len(wine))
    x_data_train = wine.iloc[:-test_split, 0:-2].as_matrix().astype(np.float32)
    y_data_train = wine.iloc[:-test_split, -1].as_matrix()
    y_data_train = np.expand_dims(y_data_train, 1).astype(np.float32)

    x_data_test = wine.iloc[-test_split:, 0:-2].as_matrix().astype(np.float32)
    y_data_test = wine.iloc[-test_split:, -1].as_matrix()
    y_data_test = np.expand_dims(y_data_test, 1).astype(np.float32)

    return x_data_train, y_data_train, x_data_test, y_data_test


def load_mnist():
    """
    Downloads the MNIST dataset
    :return: the mnist dataset as numpy arrays
    """
    download_list = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    for url in download_list:
        download_url(url)
    mnist_data = MNIST(get_data_location(), gz=True)

    train_img, train_labels = mnist_data.load_training()
    test_img, test_labels = mnist_data.load_testing()

    train_img = np.asarray(train_img).reshape(-1, 28, 28).astype(np.float32)
    train_labels = np.eye(10)[train_labels]
    test_img = np.asarray(test_img).reshape(-1, 28, 28).astype(np.float32)
    test_labels = np.eye(10)[test_labels]

    return train_img, train_labels, test_img, test_labels


def _get_wordnet_pos(treebank_tag):
    """
    Converts a treebank POS tag to wordnet tag
    :param treebank_tag:
    :return:
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def tokenise(text, split_sentence=True, clean=True):
    """
    Tokenises IMDB movie review text
    :param text: the review
    :param split_sentence: whether tokenisation of the sentence is necessary
    :param clean: whether to clean out HTML tags and encode to UTF-8 as necessary
    :return: a list of tokens
    """
    if clean:
        try:
            text = text.decode('utf-8')
        except AttributeError:
            pass

        if '<' in text:
            text = ''.join(BeautifulSoup(text, "html5lib").findAll(text=True))

    if split_sentence:
        for s in sentence_tokeniser.tokenize(text):
            for w in tokenise(s, split_sentence=False, clean=False):
                yield w
    else:
        for w, pos in nltk.pos_tag(word_tokenize(text)):
            # Skip tokens that are either contain digits or contain no alpha characters
            if any(c.isdigit() for c in w) or all(not c.isalpha() for c in w):
                continue
            wnp = _get_wordnet_pos(pos)
            if wnp is None:
                yield stemmer.stem(w).lower()
            else:
                yield lemmatiser.lemmatize(w, wnp).lower()


def load_imdb_data(dataset, batch_size=None, shuffle=True, encode=False, use_default_vocab=True, max_len=None,
                   cache_vocab=True):
    """
    Loads the IMDB dataset - note: testing and training datasets have been preprocessed into TFRecord files
    :param dataset: one of "train", "test" or "vocab" (a list of unqiue tokens)
    :param batch_size: size of dataset batch
    :param shuffle: whether to first shuffle the dataset
    :param encode: whether to encode the strings to vocab indices
    :param use_default_vocab: whether to use the vocab that is included with this package or to generate a new oen
    :param max_len: maximum string length to return
    :param cache_vocab: whether cached vocabulary  should be used
    :return: the requested dataset
    """
    global IMDB_VOCAB
    global IMDB_ALL_FILES
    global IMDB_TRAIN_FILES
    global IMDB_TEST_FILES
    if not cache_vocab:
        IMDB_VOCAB = None

    dataset = dataset.lower()
    assert dataset in ['train', 'test', 'vocab'], 'Dataset must be one of "train", "test", or "vocab".'

    if batch_size is not None:
        if dataset == 'vocab':
            vocab_batch = []
            for w in load_imdb_data('vocab'):
                vocab_batch.append(w)
                if len(vocab_batch) == batch_size:
                    yield vocab_batch
                    vocab_batch = []
            if len(vocab_batch) > 0:
                yield vocab_batch
        elif dataset in ['train', 'test']:
            text = []
            lengths = []
            ratings = []
            if encode:
                assert max_len is not None, "max_len must be set if encode=True and batch_size is not None"
                for t, l, r in load_imdb_data(dataset, None, shuffle, encode, use_default_vocab, max_len):
                    text.append(t)
                    lengths.append(l)
                    ratings.append(r)
                    if len(text) == batch_size:
                        yield np.stack(text), np.array(lengths), np.array(ratings)
                        text = []
                        lengths = []
                        ratings = []
                if len(text) > 0:
                    yield np.stack(text), np.array(lengths), np.array(ratings)
            else:
                for t, r in load_imdb_data(dataset, None, shuffle, encode, use_default_vocab, max_len):
                    text.append(t)
                    ratings.append(r)
                    if len(text) == batch_size:
                        yield text, ratings
                        text = []
                        ratings = []
                if len(text) > 0:
                    yield text, ratings
    else:
        imdb_tar = download_url("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        with tarfile.open(imdb_tar, 'r:gz') as tar:
            if IMDB_ALL_FILES is None:
                IMDB_ALL_FILES = [f for f in tar.getmembers() if f.isfile()]
                IMDB_TRAIN_FILES = [f for f in IMDB_ALL_FILES if
                                    (f.name.startswith("aclImdb/train/pos") or f.name.startswith("aclImdb/train/neg"))]
                IMDB_TEST_FILES = [f for f in IMDB_ALL_FILES if
                                   f.name.startswith("aclImdb/test/pos") or f.name.startswith("aclImdb/test/neg")]
                IMDB_ALL_FILES = IMDB_TRAIN_FILES + IMDB_TEST_FILES

            if dataset == 'vocab':
                vocab_path = os.path.join(get_data_location(), 'imdb_vocab.dat')

                if IMDB_VOCAB is None:
                    try:
                        with open(get_data_location('imdb_vocab.dat', prefer_cache=(not use_default_vocab)),
                                  'r') as vocab_file:
                            vocab = sorted([line.rstrip() for line in vocab_file])
                    except FileNotFoundError:
                        vocab = {}
                        for datafile in IMDB_ALL_FILES:
                            f = tar.extractfile(datafile)
                            for w in tokenise(f.read()):
                                vocab[w] = vocab.get(w, 0) + 1
                        vocab = sorted([k for (k, v) in vocab.items() if v > 50])
                        with open(vocab_path, 'w') as vocab_file:
                            vocab_file.writelines([line + '\n' for line in vocab])
                    vocab_dict = {}
                    for i, v in enumerate(vocab):
                        vocab_dict[v] = i + 1
                    IMDB_VOCAB = vocab_dict
                else:
                    vocab_dict = IMDB_VOCAB
                yield vocab_dict
            else:
                if dataset == 'train':
                    dataset = IMDB_TRAIN_FILES
                else:
                    dataset = IMDB_TEST_FILES
                if shuffle:
                    random.shuffle(dataset)
                for datafile in dataset:
                    f = tar.extractfile(datafile)
                    rating = int(datafile.name.split("_")[-1].split('.')[0])
                    text = f.read()
                    if encode:
                        text = encode_text(text)
                        text_length = len(text)
                        if max_len is not None:
                            if text_length > max_len:
                                text = text[:max_len]
                                text_length = max_len
                            else:
                                pad_size = max_len - text_length
                                text = np.pad(text, (0, pad_size), 'constant')
                            text_length = text_length if text_length <= max_len else max_len
                        r = np.zeros((11,))
                        r[rating] = 1
                        yield text, text_length, r
                    else:
                        yield text, rating


def get_imdb_vocab():
    """Shortcut for getting the IMDB vocabulary"""
    return next(load_imdb_data('vocab'))


def get_imdb_pretrained():
    """Returns the location of the pretrained IMDB model"""
    try:
        return get_data_location('imdb_model_pretrained', prefer_cache=True, isdir=True)
    except FileNotFoundError:
        imdb_model_path = get_data_location('imdb_model', prefer_cache=False, isdir=True)
        copy_dst = get_data_location('imdb_model_pretrained', prefer_cache=True, isdir=True, raise_if_not_found=False)
        shutil.copytree(imdb_model_path, copy_dst)
        return copy_dst


def encode_text(text, vocab=None):
    """
    Converts text into token indices according to the vocabulary
    :param text:
    :param vocab:
    :return:
    """
    if vocab is None:
        vocab = next(load_imdb_data('vocab'))
    result = []
    for w in tokenise(text):
        result.append(vocab.get(w, 0))
    return np.array(result)


def imdb_to_tfrecord(dataset, output_file):
    """
    Stores the requested IMDB data to TFRecord files
    :param dataset: the requested dataset
    :param output_file: the file name to use to store the TFRecord data (excluding extension)
    """
    assert dataset in ['test', 'train'], "Only test and train datasets can be stored"
    with tf.python_io.TFRecordWriter(
            get_data_location('%s.tfrecords' % output_file, raise_if_not_found=False)) as writer:
        for text, length, rating in load_imdb_data(dataset, shuffle=False, encode=True):
            imdb_example = tf.train.Example(features=tf.train.Features(feature={
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=text.astype(np.int64))),
                'rating': tf.train.Feature(int64_list=tf.train.Int64List(value=rating.astype(np.int64)))
            }))
            writer.write(imdb_example.SerializeToString())


def do_convolution(kernel, image):
    """
    Performs a 2D convolution over 2D data with a stride of 1
    :param kernel: convolution kernel
    :param image: input image
    :return: the convolved image
    """
    with tf.Graph().as_default() as g:
        input_image = tf.placeholder(dtype=tf.float32, shape=(None, None))
        img = tf.expand_dims(input_image, axis=0)
        img = tf.expand_dims(img, axis=-1)

        input_conv_kernel = tf.placeholder(dtype=tf.float32, shape=(None, None))
        conv_kernel = tf.expand_dims(input_conv_kernel, axis=-1)
        conv_kernel = tf.expand_dims(conv_kernel, axis=-1)

        output = tf.nn.conv2d(img, conv_kernel, (1, 1, 1, 1), 'VALID')

    with tf.Session(graph=g) as sess:
        return np.squeeze(sess.run(output, feed_dict={input_image: image,
                                                      input_conv_kernel: kernel}))


def count_parameters():
    """
    Counts the number of trainable parameters in the current grapg
    :return:
    """
    return int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
