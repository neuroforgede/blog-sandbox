import numpy as np

import deepbuzz.config as config


def generate_dataset(dec):
    data = []
    target = []

    for i in dec:
        binary = binary_of_int(i)
        label = label_of_int(i)
        data.append(binary)
        target.append(label)
    return np.array(data), np.array(target)


def binary_of_int(n):
    binary = []
    for i in range(config.NUM_DIGITS_INPUT):
        binary.append(n >> i & 1)
    return binary


def label_of_int(n):
    if n % 3 == 0 and n % 5 == 0:
        return [0, 0, 0, 1]  # fizzbuzz
    elif n % 5 == 0:
        return [0, 0, 1, 0]  # buzz
    elif n % 3 == 0:
        return [0, 1, 0, 0]  # fizz
    else:
        return [1, 0, 0, 0]  # other number


def class_of_label(label):
    if label == [3]:
        return 'fizzbuzz'
    elif label == [2]:
        return 'buzz'
    elif label == [1]:
        return 'fizz'
    elif label == [0]:
        return 'other number'


def count_fizz(data):
    count = 0
    for n in data:
        if n % 3 == 0:
            count += 1
    return count


def count_buzz(data):
    count = 0
    for n in data:
        if n % 5 == 0:
            count += 1
    return count


def count_fizzbuzz(data):
    count = 0
    for n in data:
        if n % 15 == 0:
            count += 1
    return count


def count_other_number(data):
    count = 0
    for n in data:
        if not (n % 5 == 0 or n % 3 == 0):
            count += 1
    return count
