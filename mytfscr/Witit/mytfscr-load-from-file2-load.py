#https://www.tensorflow.org/tutorials/load_data/images
#https://www.tensorflow.org/tutorials/images/classification


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model

#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')

#dot_img_file = '/tmp/model_1.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

img_height = 180
img_width = 180

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

class_names = ['daisy', 'dandelion']

#print("This image is %.2f percent daisy and %.2f percent dandelion." % (100 * (1 - score[0]), 100 * score[0]))
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



