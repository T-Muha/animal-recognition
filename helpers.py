import tensorflow as tf
import numpy as np
import PIL
import pathlib



# resizes PIL image to the given width and height
def resize_image(image, width, height):
    # Crop the image to a square with center as anchor, then resize.
    # Alternatively, could pad with zeros or resize without making square but:
    # cropping removes info, zero padding could create unintended patterns,
    # resizing without cropping or padding could distort features
    imWidth = image.size[0]
    imHeight = image.size[1]
    if width > height:
        image.crop((imWidth/2-imHeight/2, imHeight, imWidth/2+imHeight/2, 0))
    elif width < height:
        image.crop((0, imHeight/2+imWidth/2, imWidth, imHeight/2-imWidth/2))
    image.resize((width, height), PIL.Image.ANTIALIAS)
    return image

# Gets the length of a dataset since there isn't a method for that
def dataset_size(dataset):
    return tf.data.experimental.cardinality(dataset).numpy()