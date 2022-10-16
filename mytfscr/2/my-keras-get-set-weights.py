import numpy as np
import tensorflow as tf
from tensorflow import keras

logdir = "C:\\Users\\pierluigi.sicuro\\Desktop\\tf.log"

inputs = keras.Input(shape=(None, None, 2))
x=inputs
tf.shape(x)

from tensorflow.keras import layers
x = layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu")(x)
tf.shape(x)
outputs = x
tf.shape(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

{v: i for i, v in enumerate(model.layers)}



# print("Weights and biases of the layers after training the model: \n")
# for layer in model.layers:
#   print(layer.name)
#   print("Weights")
#   print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
#   print("Bias")
#   print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')

# first_layer_weights = model.layers[0].get_weights()[0]
# first_layer_biases  = model.layers[0].get_weights()[1]

layer_1_weights = model.layers[1].get_weights()[0]
layer_1_biases  = model.layers[1].get_weights()[1]

print(layer_1_weights.shape)
print(layer_1_weights)

print(layer_1_biases.shape)
print(layer_1_biases)


print(model.layers[1].get_weights())

print("Weights")
print(model.layers[1].get_weights()[0].shape)
print(model.layers[1].get_weights()[0])

print("Bias")
print(model.layers[1].get_weights()[1].shape)
print(model.layers[1].get_weights()[1])

# for layer in model.layers: print(layer.get_weights())



# Setting new weights and biases

# model.layers[i].set_weights(listOfNumpyArrays)    
# model.get_layer(layerName).set_weights(...)
# model.set_weights(listOfNumpyArrays)

model.layers[1].set_weights([np.zeros(model.layers[1].get_weights()[0].shape), np.zeros(model.layers[1].get_weights()[1].shape)])
model.layers[1].set_weights([np.random.randn(*model.layers[1].get_weights()[0].shape), np.random.randn(*model.layers[1].get_weights()[1].shape)])
model.layers[1].set_weights([np.ones(model.layers[1].get_weights()[0].shape), np.ones(model.layers[1].get_weights()[1].shape)])
print(model.layers[1].get_weights())



# for layer in model.layers: print(layer.get_config(), layer.get_weights())



# dataset = np.random.randint(0, 256, size=(1, 5, 5, 2)).astype("float32")
dataset = np.ones(shape=(1, 5, 5, 2)).astype('int')

print(dataset.shape)
print(dataset)

processed_data = model(dataset)
print(processed_data.shape)
print(processed_data)



print(model.layers[1].output)



# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),loss=keras.losses.CategoricalCrossentropy())
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


