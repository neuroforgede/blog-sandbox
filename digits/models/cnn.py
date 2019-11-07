import numpy
from keras import layers, engine
from sklearn import datasets

from digits import config


def get_model():
    # load prepared data set containing 1797 digits as 8x8 images
    digit_features, digit_classes = datasets.load_digits(n_class=config.NUM_DIGITS, return_X_y=True)
    num_samples = digit_classes.shape[0]

    # normalize features, see documentation of sklearn.datasets.load_digits!
    digit_features /= config.MAX_FEATURE

    # we need so called "one-hot" vectors
    digit_labels = numpy.zeros(shape=(num_samples, config.NUM_DIGITS))
    for index, digit_class in enumerate(digit_classes):
        digit_labels[index][digit_class] = 1.

    # we need a layer that acts as input.
    # shape of that input has to be known and depends on data.
    input_layer = layers.Input(shape=(config.NUM_FEATURES,))

    # data is 1D, convert it to its original 2D image form, which has 3 dimensions
    # remember the scalar for pixel values, which is the third dimension
    # color images dont have a scalar, but three values in this dimension instead
    hidden_layer = layers.Reshape(target_shape=(8, 8, 1))(input_layer)

    # convolutions are filters, which are useful for finding features (eg. edges)
    # 2D-convolutions take 3D data (see above) and introduce a new dimension:
    # the filtered data
    hidden_layer = layers.Conv2D(filters=64, kernel_size=(3, 3))(hidden_layer)
    hidden_layer = layers.Conv2D(filters=32, kernel_size=(1, 1))(hidden_layer)
    hidden_layer = layers.Conv2D(filters=16, kernel_size=(1, 1))(hidden_layer)

    # convert it back to 1D data, so we can map it to the 1D output
    hidden_layer = layers.Flatten()(hidden_layer)

    # last layer represents output.
    # activation of each neuron corresponds to the models decision of
    # choosing that class.
    # softmax ensures that all activations summed up are equal to 1.
    # this lets one interpret that output as a probability
    output_layer = layers.Dense(units=config.NUM_DIGITS, activation='softmax')(hidden_layer)

    # actual creation of the model with in- and output layers
    model = engine.Model(inputs=[input_layer], outputs=[output_layer])

    # transform into a trainable model by specifying the optimizing function
    # (here stochastic gradient descent),
    # as well as the loss (eg. how big of an error is produced by the model)
    # track the model's accuracy as an additional metric (only possible for classification)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
