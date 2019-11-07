from numpy import asarray

from digits.models import cnn, mlp


def image_to_ndarray(image):
    """
    Simple utility to convert an image to a array with shape (1, #pixels in image)
    :param image: image to convert
    :return: an array with shape (1, #pixels in image), converts image to greyscale
    """
    pixels = list(image.getdata())
    pixels = pixels_to_greyscale(pixels)
    return asarray(pixels)


def pixels_to_greyscale(data):
    if len(data) > 0:
        pixel = data[0]
        try:
            channels = len(pixel)
        except TypeError:
            # has no attribute __len__, therefore no tuple, so only greyscale
            channels = 1
    else:
        return data
    if channels == 1:
        return data
    for index, pixel in enumerate(data):
        data[index] = ((pixel[0] + pixel[1] + pixel[2]) / 3) * pixel[4] if channels == 4 else 1
    return data
