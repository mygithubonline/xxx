'''
# Get the working directory path
#import os
#current_dir = os.getcwd()
current_dir = "C://Users//pierluigi.sicuro//Desktop//ds//"

# Import mnist data stored in the following path: current directory -> mnist.npz
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'mnist.npz')
'''

#https://www.tensorflow.org/tutorials/load_data/numpy

import numpy as np
import tensorflow as tf

#DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
DATA_URL = 'C://Users//pierluigi.sicuro//Desktop//ds//mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']


print("Ciao!")
#print(np.shape(train_examples))
print(train_examples.shape)
print(train_labels.shape)
print(test_examples.shape)
print(test_labels.shape)

print(train_labels[0])
#print(train_examples[0])


import matplotlib.pyplot as plt
imgplot = plt.imshow(train_examples[0])
plt.show()



'''
print("train_examples.shape: " + ''.join(train_examples.shape))
print("train_labels.shape: " + train_labels.shape)
print("test_examples.shape: " + test_examples.shape)
print("test_labels.shape" + test_labels.shape)
'''


'''
#Load NumPy arrays with tf.data.Dataset
#Assuming you have an array of examples and a corresponding array of labels, 
#pass the two arrays as a tuple into tf.data.Dataset.from_tensor_slices to create a tf.data.Dataset.

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

#Use the datasets
#Shuffle and batch the datasets

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#Build and train a model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)

'''



