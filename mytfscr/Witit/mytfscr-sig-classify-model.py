#https://www.tensorflow.org/tutorials/load_data/images
#https://www.tensorflow.org/tutorials/images/classification



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
'''
Tested on tf 0.12 and 1.0
In details,
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import numpy as np
#import PIL
#import PIL.Image

import tensorflow as tf
print(tf.__version__)

#import tensorflow_datasets as tfds

import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)

#data_dir = "C://Users//pierluigi.sicuro//.keras//datasets//flower_photos"
data_dir = "C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-training"
data_dir = pathlib.Path(data_dir)

print("data_dir --> " + str(data_dir))
image_count = len(list(data_dir.glob('*/*.png')))
print("image_count --> " + str(image_count))


batch_size = 32
img_height = 180
img_width = 180


'''
use 80% of the images for training, and 20% for validation.
'''
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  #image_size=(img_width, img_height),
  batch_size=batch_size)

#print("train_ds --> " + str(train_ds.get_shape()))
'''
print("train_ds --> " + str(train_ds.image_size[0]))
print("train_ds --> " + str(train_ds.image_size[1]))
print("train_ds --> " + str(train_ds.num_channels))
'''

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  #image_size=(img_width, img_height),
  batch_size=batch_size)


#You can find the class names in the class_names attribute on these datasets.
#print("train_ds --> " + str(train_ds))
class_names = train_ds.class_names
print("class_names --> " + str(class_names))

#print("train_ds.shape --> " + str(train_ds.shape))


#Visualize the data
#import matplotlib.pyplot as plt
'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
'''

'''
You can train a model using these datasets by passing them to model.fit (shown later in this tutorial). 
If you like, you can also manually iterate over the dataset and retrieve batches of images
'''

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


'''
Standardize the data
The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; 
in general you should seek to make your input values small. 
Here, we will standardize values to be in the [0, 1] by using a Rescaling layer.
'''
from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
#normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

'''
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
'''

'''
you can include the layer inside your model definition:
Configure the dataset for performance
Let's make sure to use buffered prefetching so we can 
yield data from disk without having I/O become blocking. 
These are two important methods you should use when loading data.
.cache() keeps the images in memory after they're loaded off disk during the first epoch. 
This will ensure the dataset does not become a bottleneck while training your model. 
If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
.prefetch() overlaps data preprocessing and model execution while training.
'''
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



num_classes = 2

'''
model = tf.keras.Sequential([

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
'''


model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


'''
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_height, img_width, 3)),
    #tf.keras.layers.Flatten(input_shape=(img_width, img_height, 1)),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(128, activation='tanh'),
    #tf.keras.layers.Dense(128, activation='sigmoid'),
    #tf.keras.layers.Dense(num_classes)

    #tf.keras.layers.Flatten(input_shape=(img_height * img_width,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')

    #network.add(layers.Dense(512, activation='relu', input_shape=(img_height * img_width,)))
    #network.add(layers.Dense(10, activation='softmax'))
])
'''



model.compile(
  optimizer='adam',
  #optimizer=keras.optimizers.Adam(1e-3),
  #loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
  #loss="binary_crossentropy",
  metrics=["accuracy"]
)


model.summary()





# Load the TensorBoard notebook extension.
#%load_ext tensorboard

#from datetime import datetime
#from packaging import version

#import tensorflow as tf
#from tensorflow import keras


#print("TensorFlow version: ", tf.__version__)
#assert version.parse(tf.__version__).release[0] >= 2, \
#    "This notebook requires TensorFlow 2.0 or above."

'''
import tensorboard
tensorboard.__version__

# Clear any logs from previous runs
#rm -rf ./logs/


# Define the Keras TensorBoard callback.
logdir="logs//fit" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
'''


epochs = 3
history_model = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=[tensorboard_callback]
)

modelfilename="sig-classyfy-model-5"
from tensorflow.keras.models import load_model
model.save(modelfilename+'.h5')

historymodelpath="C://Users//pierluigi.sicuro//Desktop//Python//scr//my1//xxx//history-"
import json
# Get the dictionary containing each metric and the loss for each epoch
history_dict = history_model.history
# Save it under the form of a json file
json.dump(history_dict, open(historymodelpath+modelfilename+".json", 'w'))

