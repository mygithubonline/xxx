# C:\Users\pierluigi.sicuro>cd C:\Users\pierluigi.sicuro\Desktop\mypy1
# C:\Users\pierluigi.sicuro\Desktop\mypy1>myvenv\Scripts\activate
# (myvenv) C:\Users\pierluigi.sicuro\Desktop\mypy1>python Image_classification_from_scratch.py



######################################################################



tf.ones((2, 2,))
# <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
# array([[1., 1.],
#        [1., 1.]], dtype=float32)>
tf.ones((2, 2,)).numpy()
# array([[1., 1.],
#        [1., 1.]], dtype=float32)



######################################################################
# Axis

# Your data has some shape (19,19,5,80). This means:

# Axis 0 = 19 elements
# Axis 1 = 19 elements
# Axis 2 = 5 elements
# Axis 3 = 80 elements

# Now, negative numbers work exactly like in python lists, in numpy arrays, etc. 
# Negative numbers represent the inverse order:

# Axis -1 = 80 elements
# Axis -2 = 5 elements
# Axis -3 = 19 elements
# Axis -4 = 19 elements

# When you pass the axis parameter to the argmax function, the indices returned will be based on this axis. 
# Your results will lose this specific axes, but keep the others.

# See what shape argmax will return for each index:

# K.argmax(a,axis= 0 or -4) returns (19,5,80) with values from 0 to 18
# K.argmax(a,axis= 1 or -3) returns (19,5,80) with values from 0 to 18
# K.argmax(a,axis= 2 or -2) returns (19,19,80) with values from 0 to 4
# K.argmax(a,axis= 3 or -1) returns (19,19,5) with values from 0 to 79

import numpy as np
a = np.random.randint(0, 256, size=(2,3,5,4)).astype("float32")
print(a.shape)
print(a)
print(tf.keras.backend.argmax(a, axis=-1))
print(tf.keras.backend.max(a, axis=-1))

# >>> a = np.random.randint(0, 256, size=(2,3,5,4)).astype("float32")
# >>> print(a.shape)
# (2, 3, 5, 4)
# >>> print(a)
# [[[[230. 176.  52. 149.]
#    [243.  64.  46. 131.]
#    [176.  37. 176. 233.]
#    [ 96.  72.  95.  72.]
#    [249. 144. 120.  46.]]

#   [[ 13. 221. 195. 116.]
#    [160. 155. 132. 116.]
#    [203.  48. 116. 117.]
#    [ 96. 197. 141.  60.]
#    [ 89. 133.  27.  49.]]

#   [[  1.  63.  79.  26.]
#    [128.  80. 172.  86.]
#    [ 43. 104.  50.  96.]
#    [201. 227.  61. 100.]
#    [253.  57.   0. 248.]]]


#  [[[239. 214. 244.  74.]
#    [ 47. 114. 145.  51.]
#    [ 45. 242.  88. 110.]
#    [110.  95. 164.  83.]
#    [121. 188.   1. 214.]]

#   [[ 36. 220. 130.  45.]
#    [174.  30. 126.  74.]
#    [196. 184.  29. 160.]
#    [119. 190.  28. 176.]
#    [245.  21. 151. 188.]]

#   [[140.  90.  18.   2.]
#    [168. 240. 169.  24.]
#    [254.   1.  76. 100.]
#    [125.  11. 146. 194.]
#    [154.  55. 181. 100.]]]]
# >>> print(tf.keras.backend.argmax(a, axis=-1))
# tf.Tensor(
# [[[0 0 3 0 0]
#   [1 0 0 1 1]
#   [2 2 1 1 0]]

#  [[2 2 1 2 3]
#   [1 0 0 1 0]
#   [0 1 0 3 2]]], shape=(2, 3, 5), dtype=int64)
# >>> print(tf.keras.backend.max(a, axis=-1))
# tf.Tensor(
# [[[230. 243. 233.  96. 249.]
#   [221. 160. 203. 197. 133.]
#   [ 79. 172. 104. 227. 253.]]

#  [[244. 145. 242. 164. 214.]
#   [220. 174. 196. 190. 245.]
#   [140. 240. 254. 194. 181.]]], shape=(2, 3, 5), dtype=float32)



######################################################################


# Create example NPZ file
import numpy as np
myarray1 = np.array([0,1,2,3])
myarray2 = np.array([4,5,2,3])
np.savez('npzfile.npz',array1=myarray1,array2=myarray2) # array1 will be the name with which you can retrieve myarray1
# Read it in
data = np.load('npzfile.npz')
print(data.files)
# ['array1', 'array2']

# use array1 key to retrieve myarray
data['array1']
# array([0, 1, 2, 3])
data['array2']
# array([4, 5, 2, 3])
print(data['array2'] )
# [4 5 2 3]


######################################################################

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# DATA_URL = 'C:\\Users\\pierluigi.sicuro\\Downloads'
# path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

path = 'C:\\Users\\pierluigi.sicuro\\Downloads\\mnist.npz'

import numpy as np
data = np.load(path)

data.files
# ['x_test', 'x_train', 'y_train', 'y_test']

xt=data['x_train']
xt.shape
# (60000, 28, 28)
x=xt[1,:,:]
x.shape
# (28, 28)


# with np.load(path) as data:
#   train_examples = data['x_train']
#   train_labels = data['y_train']
#   test_examples = data['x_test']
#   test_labels = data['y_test']
#   print("data:----------------------")
#   print(data)

######################################################################

import tensorflow as tf
from tensorflow import keras

dataset = keras.preprocessing.image_dataset_from_directory("C:\\Users\\pierluigi.sicuro\\Desktop\\ds\\flower_photos", batch_size=64, image_size=(200, 200))
# Found 3670 files belonging to 5 classes.

dataset.cardinality()
# <tf.Tensor: shape=(), dtype=int64, numpy=58>
dataset.cardinality().numpy()
# 58

tf.data.experimental.cardinality(dataset)
# <tf.Tensor: shape=(), dtype=int64, numpy=58>

print(57*64-3670)
# -22

dataset.take(1)
# <TakeDataset element_spec=(TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

i=0
for data, labels in dataset:
  i=i+1
  if i == 58:
    break

print(data.shape)
# (22, 200, 200, 3)
print(labels.shape)
# (22,)



def gen():
  i=0
  for x, y in dataset:
    i=i+1
    if i == 2:
      return x, y


[data, labels]= gen()
print(data.shape)
print(labels.shape)




# # For demonstration, iterate over the batches yielded by the dataset.
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32

for x, y in dataset:
   print(x.shape)  # (64, 200, 200, 3)
   print(x.dtype)  # float32
   print(y.shape)  # (64,)
   print(y.dtype)  # int32   

for x, y, z in dataset:
   print(x.shape)  # (64, 200, 200, 3)
   print(x.dtype)  # float32
   print(y.shape)  # (64,)
   print(y.dtype)  # int32   
   print(z.shape)  # (64,)
   print(z.dtype)  # int32  


######################################################################

import numpy as np
r = np.random.randint(1,100, size=(2,3,5))
r.shape
# (2, 3, 5)
print(r)
# [[[75 12 96 50 34]
#   [88 23 65 98  6]
#   [33 70 66 67 81]]

#  [[75 88 14  6 80]
#   [86 14 44 63  7]
#   [96 84 39 93 27]]]

dataset = tf.data.Dataset.from_tensor_slices(r)


######################################################################



dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)

dataset = dataset.apply(dataset_fn)
list(dataset.as_numpy_iterator())


######################################################################

dataset = tf.data.Dataset.range(7).window(3)
for window in dataset:
  print([item.numpy() for item in window])
# [0, 1, 2]
# [3, 4, 5]
# [6]

######################################################################

dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3],
                                              'b': [4, 5, 6],
                                              'c': [7, 8, 9]})
def to_numpy(ds):
  return list(ds.as_numpy_iterator())

dataset1 = dataset.window(1)
for windows in dataset1:
  print(tf.nest.map_structure(to_numpy, windows))

# {'a': [1], 'b': [4], 'c': [7]}
# {'a': [2], 'b': [5], 'c': [8]}
# {'a': [3], 'b': [6], 'c': [9]}  

dataset2 = dataset.window(2)
for windows in dataset2:
  print(tf.nest.map_structure(to_numpy, windows))

# {'a': [1, 2], 'b': [4, 5], 'c': [7, 8]}
# {'a': [3], 'b': [6], 'c': [9]}  

dataset3 = dataset.window(3)
for windows in dataset3:
  print(tf.nest.map_structure(to_numpy, windows))    

# {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}

for x, y, z in dataset3:
   print(x)
   print(y)
   print(z)
# a
# b
# c   

for x in dataset3:
  print(x)

# {'a': <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>, 
# 'b': <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>, 
# 'c': <_VariantDataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>}

# https://www.tensorflow.org/api_docs/python/tf/data/Dataset

######################################################################


# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")   

# plt.show()

######################################################################
######################################################################
######################################################################



# Example: turning strings into sequences of integer word indices

from tensorflow.keras.layers import TextVectorization
import numpy as np

# Example training data, of dtype `string`.
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# Create a TextVectorization layer instance. It can be configured to either
# return integer token indices, or a dense token representation (e.g. multi-hot
# or TF-IDF). The text standardization and text splitting algorithms are fully
# configurable.

vectorizer = TextVectorization(output_mode="int")
#one-hot encoded bigrams
#vectorizer = TextVectorization(output_mode="binary", ngrams=2)

# Calling `adapt` on an array or dataset makes the layer generate a vocabulary
# index for the data, which can then be reused when seeing new data.
vectorizer.adapt(training_data)

# After calling adapt, the layer is able to encode any n-gram it has seen before
# in the `adapt()` data. Unknown n-grams are encoded via an "out-of-vocabulary"
# token.
integer_data = vectorizer(training_data)
print(integer_data)

######################################################################

# Example: normalizing features

from tensorflow.keras.layers import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

######################################################################

# Example: rescaling & center-cropping images

# Both the Rescaling layer and the CenterCrop layer are stateless, so it isn't necessary to call adapt() in this case.

from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))


######################################################################

# Get the data as Numpy arrays

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling


# Build a simple model
inputs = keras.Input(shape=(28, 28))

tf.shape(inputs)
# <KerasTensor: shape=(3,) dtype=int32 inferred_value=[None, 28, 28] (created by layer 'tf.compat.v1.shape')>
x = layers.Rescaling(1.0 / 255)(inputs)
tf.shape(x)
# <KerasTensor: shape=(3,) dtype=int32 inferred_value=[None, 28, 28] (created by layer 'tf.compat.v1.shape_1')>
x = layers.Flatten()(x)
tf.shape(x)
# <KerasTensor: shape=(2,) dtype=int32 inferred_value=[None, 784] (created by layer 'tf.compat.v1.shape_2')>
x = layers.Dense(128, activation="relu")(x)
tf.shape(x)
# <KerasTensor: shape=(2,) dtype=int32 inferred_value=[None, 128] (created by layer 'tf.compat.v1.shape_3')>
x = layers.Dense(128, activation="relu")(x)
tf.shape(x)
# <KerasTensor: shape=(2,) dtype=int32 inferred_value=[None, 128] (created by layer 'tf.compat.v1.shape_4')>
outputs = layers.Dense(10, activation="softmax")(x)
tf.shape(outputs)
# <KerasTensor: shape=(2,) dtype=int32 inferred_value=[None, 10] (created by layer 'tf.compat.v1.shape_5')>




model = keras.Model(inputs, outputs)

model.summary()
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28)]          0

#  rescaling (Rescaling)       (None, 28, 28)            0

#  flatten (Flatten)           (None, 784)               0

#  dense (Dense)               (None, 128)               100480

#  dense_1 (Dense)             (None, 128)               16512

#  dense_2 (Dense)             (None, 10)                1290

# =================================================================
# Total params: 118,282
# Trainable params: 118,282
# Non-trainable params: 0
# _________________________________________________________________


# Compile the model
# Loss and optimizer can be specified via their string identifiers (in this case their default constructor argument values are used):
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())

batch_size = 64

# Train the model for 1 epoch from Numpy data
# print("Fit on NumPy data")
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# # Train the model for 1 epoch using a dataset
# print("Fit on Dataset")
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)


dataset = np.random.randint(0, 256, size=(1064, 28, 28)).astype("float32")
tf.shape(dataset)
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1064,   28,   28])>


processed_data = model(dataset)
print(processed_data.shape)
# (1064, 10)

numpy_array_of_samples=dataset
tf.shape(numpy_array_of_samples)
# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1064,   28,   28])>
numpy_array_of_labels=np.random.randint(0, 10, size=(1064, 10))
tf.shape(numpy_array_of_labels)
# <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1064,   10])>

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:\\Users\\pierluigi.sicuro\\Desktop\\mypy1")
history = model.fit(numpy_array_of_samples, numpy_array_of_labels,batch_size=32, epochs=10, callbacks=[tensorboard_callback])


# The fit() call returns a "history" object which records what happened over the course of training. 
# The history.history dict contains per-epoch timeseries of metrics values 
# (here we have only one metric, the loss, and one epoch, so we only get a single scalar):

print(history.history)
# {'loss': [809562.125, 871654.25, 937725.625, 1012495.875, 1084302.125, 1166174.0, 1245826.375, 1330758.625, 1418148.75, 1471925.75]}

######################################################################

# Building models with the Keras Functional API
# A "layer" is a simple input-output transformation (such as the scaling & center-cropping transformations above). 
# For instance, here's a linear projection layer that maps its inputs to a 16-dimensional feature space:

# dense = keras.layers.Dense(units=16)

# A "model" is a directed acyclic graph of layers. You can think of a model as a "bigger layer" that encompasses 
# multiple sublayers and that can be trained via exposure to data.

# The most common and most powerful way to build Keras models is the Functional API. 
# To build models with the Functional API, 
# you start by specifying the shape (and optionally the dtype) of your inputs. 
# If any dimension of your input can vary, you can specify it as None. 
# For instance, an input for 200x200 RGB image would have shape (200, 200, 3), 
# but an input for RGB images of any size would have shape (None, None, 3).

# Let's say we expect our inputs to be RGB images of arbitrary size

import numpy as np
import tensorflow as tf
from tensorflow import keras

inputs = keras.Input(shape=(None, None, 3))
tf.shape(inputs)

# Center-crop images to 150x150
from tensorflow.keras.layers import CenterCrop
x = CenterCrop(height=150, width=150)(inputs)
tf.shape(x)

# Rescale images to [0, 1]
from tensorflow.keras.layers import Rescaling
x = Rescaling(scale=1.0 / 255)(x)
tf.shape(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
tf.shape(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
tf.shape(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
tf.shape(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
tf.shape(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
tf.shape(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)
tf.shape(x)

# Add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
tf.shape(outputs)

# Once you have defined the directed acyclic graph of layers that turns your input(s) into your outputs, 
# instantiate a Model object: This model behaves basically like a bigger layer. 
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Loss and optimizer can be specified via their string identifiers (in this case their default constructor argument values are used):
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

num_date = 96

dataset = np.random.randint(0, 256, size=(num_date, 200, 200, 3)).astype("float32")

processed_data = model(dataset)
print(processed_data.shape)

numpy_array_of_samples=dataset
tf.shape(numpy_array_of_samples)

numpy_array_of_labels=np.random.randint(0, 10, size=(num_date, 10))
# numpy_array_of_labels=processed_data
tf.shape(numpy_array_of_labels)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:\\Users\\pierluigi.sicuro\\Desktop\\mypy1")

# the fit() method accepts Dataset objects, Python generators that yield batches of data, or NumPy arrays.
history = model.fit(numpy_array_of_samples, numpy_array_of_labels,batch_size=32, epochs=3, callbacks=[tensorboard_callback])

print(history.history)

model.save("C:\\Users\\pierluigi.sicuro\\Desktop\\mypy1")

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print(weights)

num_date = 1
x_test = np.random.randint(0, 256, size=(num_date, 200, 200, 3)).astype("float32")
y_test = np.random.randint(0, 10, size=(num_date, 10))

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=2)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)



######################################################################

# tensorboard --logdir="C:\\Users\\pierluigi.sicuro\\Desktop\\mypy1"

######################################################################
# Monitoring metrics
######################################################################
image_gen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, zoom_range=0.5, rotation_range=45, width_shift_range=0.15, height_shift_range=0.15)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory = train_dir, shuffle=True, target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='binary')

image_gen = ImageDataGenerator(rescale = 1./255)

val_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory = val_dir, shuffle=True, target_size=(IMG_SHAPE,IMG_SHAPE))


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(5),
                                    
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
######################################################################



