import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
from helpers import *

# Displays the given ndarray of image data in a window
def show_ndarray_images(ndImages):
    print(1/2)
    # Calculate images and pixels per side
    imgPerSide = int(np.ceil(ndImages.shape[0]**(1/2)))
    pxPerSide = int(imgPerSide * 10 / 3)
    plt.figure(figsize=(pxPerSide, pxPerSide))
    for i, ndImage in enumerate(ndImages):
        ax = plt.subplot(imgPerSide, imgPerSide, i + 1)
        imageEntry = PIL.Image.fromarray(ndImage.astype('uint8'), 'RGB')
        resizedImageEntry = np.array(resize_image(imageEntry, 200, 200))
        plt.imshow(resizedImageEntry)
        plt.axis("off")

    plt.show()

# Displays the given dataset of imgaes in a window - deprecated?
def show_dataset_images(datasetImages):
    class_names = datasetImages.class_names
    datasetSize = dataset_size(datasetImages)
    print(datasetSize)
    # plt.figure(figsize=(datasetSize, datasetSize))
    plt.figure(figsize=(10, 10))
    # take all elements of the batch_dataset to get them as a take_dataset - whats the difference?
    for images, labels in datasetImages.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            imageEntry = PIL.Image.fromarray(images[i].numpy().astype('uint8'), 'RGB')
            resizedImageEntry = np.array(resize_image(imageEntry, 200, 200))
            plt.imshow(resizedImageEntry)
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.show()