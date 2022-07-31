import tensorflow as tf
import tensorflow.keras.preprocessing.image as tfImage
import numpy as np
import PIL
import pathlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from helpers import *
from visualizers import show_ndarray_images

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())



# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# Configuration
imgHeight = 64
imgWidth = 64
batchSize = 32
epochs = 10

# Create datasets from the image directories
# TODO - implement cross-validation
dataDir = pathlib.Path("archive/Animals-10")
dataGenerator = tfImage.ImageDataGenerator(validation_split=0.3, featurewise_std_normalization=True)
trainGenerator = dataGenerator.flow_from_directory(
    dataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode='grayscale',
    subset='training')
valGenerator = dataGenerator.flow_from_directory(
    dataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode='grayscale',
    subset='validation')

# # Take a look at the generator data
batchX, batchY = trainGenerator.next()
# for x in batchX:
#     print(x.shape)
# print('Batch Shape: %s, Minimum: %.3f, Maximum: %.3f' % (batchX.shape, batchX.min(), batchX.max()))
# show_ndarray_images(batchX)color_mode='grayscale',
# batchX = np.array(map(lambda x: x.mean(), batchX))
# print(batchX.size)
# show_ndarray_images(batchX)


# !!! remember to rescale image data for the network

# Define the model
# Sequential: defines a sequence of layers in network
# Flatten: takes 2-dimensional layer and makes it 1-dimensional
# Dense: adds a layer of neurons
# Relu: activation function - return negative values as zero
# Softmax: returns a converted array where max is 1 and all other values are 0
# Pooling: Compress image by choosing maximum of every four squares
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(imgWidth,imgHeight,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compile the model
model.compile(optimizer = tf.optimizers.Adam(),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

# fit the model
model.fit(
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