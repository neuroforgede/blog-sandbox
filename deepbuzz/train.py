from os import path
from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.compat.v1 import set_random_seed
import numpy as np

from deepbuzz.model import mlp
import deepbuzz.config as config
from deepbuzz.data import generate_dataset, count_fizz, count_buzz, count_fizzbuzz, count_other_number


np.random.seed(5)


def main():
    all_numbers = [i for i in range(0, pow(2, 10))]
    features, labels = generate_dataset(all_numbers)

    total_count = len(all_numbers)
    class_weights = {
        0: total_count / count_other_number(all_numbers),
        1: total_count / count_fizz(all_numbers),
        2: total_count / count_buzz(all_numbers),
        3: total_count / count_fizzbuzz(all_numbers),
    }

    model = mlp.get_model()
    set_random_seed(42)

    run_name = 'deepbuzz-{:%d-%b_%H-%M-%S}'.format(datetime.now())
    dir_path = path.dirname(path.realpath(__file__))
    log_dir = path.join(dir_path, 'logs', run_name)
    print('logging to "{}"'.format(log_dir))
    tb_callback = callbacks.TensorBoard(log_dir=log_dir)
    model.fit(features, labels, batch_size=config.BATCH_SIZE, epochs=20000, validation_split=.2, shuffle=True, class_weight=class_weights, callbacks=[tb_callback])
    model.save('model/deepbuzz.h5')


if __name__ == "__main__":
    main()
