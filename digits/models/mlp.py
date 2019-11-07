from keras import layers, engine

from digits import config


def get_model():
    # we need a layer that acts as input.
    # shape of that input has to be known and depends on data.
    input_layer = layers.Input(shape=(config.NUM_FEATURES,))

    # hidden layers are the model's power to fit data.
    # number of neurons and type of layers are crucial.
    # idea behind decreasing number of units per layer:
    # increase the "abstraction" in each layer...
    hidden_layer = layers.Dense(units=512)(input_layer)
    hidden_layer = layers.Dense(units=256)(hidden_layer)
    hidden_layer = layers.Dense(units=128)(hidden_layer)
    hidden_layer = layers.Dense(units=64)(hidden_layer)
    hidden_layer = layers.Dense(units=32)(hidden_layer)

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
