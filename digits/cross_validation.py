# seed the random generator, for reproducible results
from numpy import random
random.seed(1337)
# same for tensorflow
import tensorflow as tf
tf.set_random_seed(42)

from datetime import datetime
from os import path

import numpy
from PIL import Image
from keras import callbacks
from sklearn import datasets

from sklearn.model_selection import KFold

from digits.models.mlp import get_model

from digits.util import image_to_ndarray
from digits.config import NUM_DIGITS, MAX_FEATURE


# directory of this file
module_dir = path.dirname(path.realpath(__file__))

# load prepared data set containing 1797 digits as 8x8 images
digit_features, digit_classes = datasets.load_digits(n_class=NUM_DIGITS, return_X_y=True)
num_samples = digit_classes.shape[0]

# normalize features, see documentation of sklearn.datasets.load_digits!
# neural networks work best with normalized data
digit_features /= MAX_FEATURE

# we need so called "one-hot" vectors
# one-hots are vectors, where all entries are 0 except the target class, which is 1
digit_labels = numpy.zeros(shape=(num_samples, NUM_DIGITS))
for index, digit_class in enumerate(digit_classes):
    digit_labels[index][digit_class] = 1.

# get a neural net, that can fit our problem and remember its initial weights
model = get_model()
initial_weights = model.get_weights()

# prints a human readable summary of the model to the out-stream
model.summary()

# initialize the cross validation folds api
kfold = KFold(3, True, 1)

run_datetime = datetime.now()
fold = 0

# iterate over all possible fold combinations
for train, test in kfold.split(digit_features):
    # split the data into features and labels depending on the fold
    train_x, train_y = digit_features[train], digit_labels[train]
    test_x, test_y = digit_features[test], digit_labels[test]

    # a callback to log loss/accuracy etc. for tensorboard to visualize
    run_name = 'digits-{:%d-%b_%H-%M-%S}-fold-{}'.format(run_datetime, fold)
    log_dir = path.join(module_dir, 'logs', run_name)
    tb_callback = callbacks.TensorBoard(log_dir=log_dir)

    # reset the model's weights
    model.set_weights(initial_weights)

    # training the model
    model.fit(
        train_x, train_y,
        batch_size=32, epochs=30, 
        validation_split=.0, 
        validation_data=(test_x, test_y),
        callbacks=[tb_callback]
    )

    fold += 1
