import tensorflow as tf
import tensorflow.keras.preprocessing.image as tfImage
import numpy as np
import PIL
import pathlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from helpers import *
from visualizers import *

# Configuration
imgHeight = 200
imgWidth = 200
batchSize = 32
epochs = 15

# Create datasets from the image directories
# TODO - implement cross-validation
dataDir = pathlib.Path("C:/Users/thmuh/Documents/Programming/archive/Animals-10")
dataGenerator = tfImage.ImageDataGenerator(validation_split=0.3)
trainGenerator = dataGenerator.flow_from_directory(
    dataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    subset='training')
valGenerator = dataGenerator.flow_from_directory(
    dataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    subset='validation')

# # Take a look at the generator data
# batchX, batchY = trainGenerator.next()
# for x in batchX:
#     print(x.shape)
# print('Batch Shape: %s, Minimum: %.3f, Maximum: %.3f' % (batchX.shape, batchX.min(), batchX.max()))
# show_ndarray_images(batchX)
# batchX = np.array(map(lambda x: x.mean(), batchX))
# print(batchX.size)
# show_ndarray_images(batchX)


# !!! remember to rescale image data for the network

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (2,2), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(800, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile the model
model.compile(optimizer = tf.optimizers.Adam(),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# fit the model
model.fit_generator(
  trainGenerator,
  steps_per_epoch= batchSize,
  validation_data= valGenerator,
  validation_steps= batchSize,
  epochs=epochs)


# def prepare_tensor(imageTensor, imgWidth, imgHeight):
#     print(imageTensor.get_shape())
#     # sess = tf.compat.v1.InteractiveSession()
#     # image = imageTensor.eval(session=sess)
#     # print(image)
#     return

# trainingData = tf.keras.preprocessing.image_dataset_from_directory(
#     dataDir,
#     validation_split=0.3,
#     subset="training",
#     seed=111,
#     image_size=(imgHeight, imgWidth),
#     batch_size=batchSize
# )

# valData = tf.keras.preprocessing.image_dataset_from_directory(
#     dataDir,
#     validation_split=0.3,
#     subset="validation",
#     seed=111,
#     image_size=(imgHeight, imgWidth),
#     batch_size=batchSize
# )

# trainingData.map(temp)
# sess = tf.compat.v1.InteractiveSession()
# trainingData.map(lambda x, y: prepare_tensor(x, imgWidth, imgHeight))

# # print(type(trainingData))
# # print(type(trainingData.take(1)))
# show_dataset_images(trainingData)

# class_names = trainingData.class_names
# plt.figure(figsize=(10, 10))
# for images, labels in trainingData.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         imageEntry = PIL.Image.fromarray(images[i].numpy().astype('uint8'), 'RGB')
#         resizedImageEntry = np.array(resize_image(imageEntry, imgWidth, imgHeight))
#         plt.imshow(resizedImageEntry)
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# plt.show()

# testImage = resize_image(trainingData.take(1), imgWidth, imgHeight)