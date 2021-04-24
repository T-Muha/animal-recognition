import tensorflow as tf

# get fashion data from mnist
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
print(type(mnist))
print(type(training_images))