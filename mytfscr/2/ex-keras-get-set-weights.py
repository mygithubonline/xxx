
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

# Create a small input dataset with output targets.

x = np.random.randn(100)
y = x*3 + np.random.randn(100)*0.8

# Create a neural network model with 2 layers.

model = Sequential()
model.add(Dense(4, input_dim = 1, activation = 'linear', name = 'layer_1'))
model.add(Dense(1, activation = 'linear', name = 'layer_2'))
model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mse'])

# Here, the first layer has 4 units(4 neurons/ 4 nodes), and the second layer has 1 unit.  The first layer takes the input 
# and the second layer gives the output. The linear activation function is used as we are making a linear regression model.

# Use the get_weights() function to get the weights and biases of the layers before training the model. 
# These are the weights and biases with which the layers will be initialized.

print("Weights and biases of the layers before training the model: \n")
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')

  # Did you notice the shape of the weights and biases? Weights of a layer are of the shape (input x units) 
  # and biases are of the shape (units,). get_weights() function returned a list consisting of Numpy arrays. 
  # Index 0 of the list has the weights array and index 1 has the bias array. The model.add(Dense()) function 
  # has an argument kernel_initializer that initializes the weights matrix created by the layer. 
  # The default kernel_initializer is glorot_uniform. Refer to the official Keras documentation on initializers 
  # for more information on glorot_uniform and other initializers. The default initial values of biases are zero.

# Fit the model and see the newly updated weights after training the model.

model.fit(x,y, batch_size = 1, epochs = 10, shuffle = False)

print("Weights and biases of the layers after training the model: \n")
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')

# Let us plot and see how well our linear line fits the model.

plt.figure(figsize = (8,8))
plt.plot(x,y,'o',x,model.predict(x),'g')
plt.show()

# The weights passed to the set_weights() function, as mentioned earlier, must be of the same shape as returned by get_weights().

# Setting new weights and biases
for layer in model.layers:
  a,b = layer.get_weights()[0].shape
  layer.set_weights([np.random.randn(a,b), np.ones(layer.get_weights()[1].shape)])

# This part of the code might seem confusing. Let me explain. In the line, a,b = layer.get_weights()[0].shape 
# we are extracting the shape tuple of the weights array given by get_weights()[0] in separate variables a and b. 
# In the last line, we pass a list of NumPy arrays – first is an array with shape (a,b) for weights and second 
# is an array with shape corresponding to the bias array, or to say, the last line is equal to layer.set_weights([weights_array, bias_array]).

print("Weights and biases of the layers after setting the new weights and biases: \n")
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')


  # Next, train the model again with the newly set weights and then see the newly updated weights after training the model.

model.fit(x,y, batch_size = 1, epochs = 10, shuffle = False)
print("Weights and biases of the layers after training the model with new weights and biases: \n")
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')


# Finally, plot this new model.

plt.figure(figsize = (8,8))
plt.plot(x,y,'o',x,model.predict(x),'g')
plt.show()




