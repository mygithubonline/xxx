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
data_dir = "C://Users//pierluigi.sicuro//Desktop//ds//1"
data_dir = pathlib.Path(data_dir)

print("data_dir --> " + str(data_dir))
image_count = len(list(data_dir.glob('*/*.jpg')))
print("image_count --> " + str(image_count))


'''
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))
print("len(roses) --> " + str(len(roses)))
'''


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
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



#You can find the class names in the class_names attribute on these datasets.
#print("train_ds --> " + str(train_ds))
class_names = train_ds.class_names
print("class_names --> " + str(class_names))



#Visualize the data
import matplotlib.pyplot as plt
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
'''
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
'''

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
    tf.keras.layers.Flatten(input_shape=(180, 180, 3)),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes)
])



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
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=[tensorboard_callback]
)

#from tensorflow.keras.models import load_model
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
#model = load_model('my_model.h5')

'''
Visualize training results
Create plots of loss and accuracy on the training and validation sets
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show()




#testImage = "C://Users//pierluigi.sicuro//Desktop//ds//newcases//15987457_49dc11bf4b.jpg"
#testImage = "C://Users//pierluigi.sicuro//Desktop//ds//newcases//134409839_71069a95d1_m.jpg"
testImage = "C://Users//pierluigi.sicuro//Desktop//ds//newcases//138166590_47c6cb9dd0.jpg"

img = tf.keras.preprocessing.image.load_img(
    #"PetImages/Cat/6779.jpg", target_size=image_size
    testImage
    , target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
#img_array = np.array([img_array])  # Convert single image to a batch.
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
print(img_array.shape)

predictions = model.predict(img_array)
#predictions = model.predict_on_batch(img_array)
print(predictions)

#score = predictions[0]
score = tf.nn.softmax(predictions[0])
print(score.shape)
print(score)

#print("This image is %.2f percent daisy and %.2f percent dandelion." % (100 * (1 - score[0]), 100 * score[0]))
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



'''
#Using tf.data for finer control

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
  print(f.numpy())



class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)



val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)



print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())



def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label



# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())



def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)



image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")



model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

'''




'''
#Using TensorFlow Datasets

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)



num_classes = metadata.features['label'].num_classes
print(num_classes)



get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))



train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

'''
