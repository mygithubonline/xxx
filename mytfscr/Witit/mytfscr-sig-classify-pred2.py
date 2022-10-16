import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model

modelfilename="sig-classyfy-model-4"
historymodelpath="C://Users//pierluigi.sicuro//Desktop//Python//scr//my1//xxx//history-"

model = load_model(modelfilename+'.h5')

#dirname="C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-signed-test//"
dirname="C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-unsigned-test//"

def predict_model(filename):

    img_height = 180
    img_width = 180

    testImage = dirname+filename

    img = tf.keras.preprocessing.image.load_img(
        #"PetImages/Cat/6779.jpg", target_size=image_size
        testImage
        , target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = np.array([img_array])  # Convert single image to a batch.
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    #print(img_array.shape)

    predictions = model.predict(img_array)
    #predictions = model.predict_on_batch(img_array)
    #print(predictions)
    #score = predictions[0]
    score = tf.nn.softmax(predictions[0])
    #print(score.shape)
    print(score)

    class_names = ['signed', 'unsigned']

    #print("This image is %.2f percent daisy and %.2f percent dandelion." % (100 * (1 - score[0]), 100 * score[0]))
    print(
        "{} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    print()


'''
filenames = os.listdir(dirname)
for x in range(len(filenames)): 
    print(filenames[x])
    predict_model(filenames[x])
'''

import json
history = json.load(open(historymodelpath+modelfilename+".json", 'r'))

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

#print(acc[0])

epochs = 3

epochs_range = range(epochs)


import matplotlib.pyplot as plt
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
plt.show()



