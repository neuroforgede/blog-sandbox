from tensorflow.keras import layers, Model

from deepbuzz import config


def get_model():
   
    input_layer = layers.Input(shape=(config.NUM_DIGITS_INPUT,))

    hidden_layer = layers.Dense(units=128, activation='relu')(input_layer)
    hidden_layer = layers.Dropout(0.3)(hidden_layer)

    hidden_layer = layers.Dense(units=128, activation='relu')(hidden_layer)
    hidden_layer = layers.Dropout(0.2)(hidden_layer)

    output_layer = layers.Dense(units=config.NUM_CLASSES_OUTPUT, activation='softmax')(hidden_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
