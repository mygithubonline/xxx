import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

logdir = "C:\\Users\\pierluigi.sicuro\\Desktop\\tf.log"



# model = keras.Sequential()
# model.add(keras.Input(shape=(None, None, 3)))  # 250x250 RGB images
# model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))

# # Can you guess what the current output shape is at this point? Probably not.
# # Let's just print it:
# model.summary()

# # The answer was: (40, 40, 32), so we can keep downsampling...

# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(2))

# # And now?
# model.summary()

# # Now that we have 4x4 feature maps, time to apply global max pooling.
# model.add(layers.GlobalMaxPooling2D())

# # Finally, we add a classification layer.
# model.add(layers.Dense(10))







inputs = keras.Input(shape=(None, None, 3))
x=inputs
tf.shape(x)

# Center-crop images to 150x150
from tensorflow.keras.layers import CenterCrop
x = CenterCrop(height=150, width=150)(x)
tf.shape(x)

# Rescale images to [0, 1]
from tensorflow.keras.layers import Rescaling
x = Rescaling(scale=1.0 / 255)(x)
tf.shape(x)

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
num_classes = 1
x = layers.Dense(num_classes, activation="softmax")(x)

outputs = x
tf.shape(outputs)

# Once you have defined the directed acyclic graph of layers that turns your input(s) into your outputs, 
# instantiate a Model object: This model behaves basically like a bigger layer. 
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()



# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())
# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")


num_date = 96
numpy_array_of_samples = np.random.randint(0, 256, size=(num_date, 200, 200, 3)).astype("float32")
tf.shape(numpy_array_of_samples)

numpy_array_of_labels=np.random.uniform(0, 1, size=(num_date, 1))
tf.shape(numpy_array_of_labels)

# numpy_array_of_labels = model(numpy_array_of_samples)
# print(numpy_array_of_labels.shape)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# the fit() method accepts Dataset objects, Python generators that yield batches of data, or NumPy arrays.
history = model.fit(numpy_array_of_samples, numpy_array_of_labels,batch_size=32, epochs=3, callbacks=[tensorboard_callback])

print(history.history)

# model.save(logdir)

# for layer in model.layers:
#     weights = layer.get_weights() # list of numpy arrays
#     print(weights)

num_date = 1
x_test = np.random.randint(0, 256, size=(num_date, 200, 200, 3)).astype("float32")
y_test = np.random.uniform(0, 1, size=(num_date, 1))

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

