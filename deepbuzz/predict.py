import numpy as np
from tensorflow.keras.models import load_model

import deepbuzz.data as data


def main():
    # returns a compiled model identical to the trained one of solution.py
    model = load_model('model/deepbuzz.h5', compile=True)

    user_input = int(input("Input an integer "))

    # binary of input
    binary_number = data.binary_of_int(user_input)

    # prediction
    prediction = model.predict(np.array([binary_number]))

    # get class label of argmax
    prediction = data.class_of_label(prediction.argmax(axis=-1))
    print(f'The number {user_input} is category: {prediction}')


if __name__ == "__main__":
    main()
