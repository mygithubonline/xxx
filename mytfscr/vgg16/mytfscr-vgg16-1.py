#C:\Users\pierluigi.sicuro>cd C:\Users\pierluigi.sicuro\Desktop\Python\scr\my1 & venv\Scripts\activate
#(venv) C:\Users\pierluigi.sicuro\Desktop\Python\scr\my1>python C:\Users\pierluigi.sicuro\Desktop\Python\TensorFlow\mytfscr\vgg16\mytfscr-vgg16-1.py
from tensorflow.keras.applications.vgg16 import VGG16
model_conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))

model_conv_base.summary()

#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#base_dir = 'C://Users//pierluigi.sicuro//Desktop//ds//flower_photos'
#train_dir = os.path.join(base_dir, 'flower_photos')
train_dir = 'C://Users//pierluigi.sicuro//Desktop//ds//flower_photos'
#validation_dir = os.path.join(base_dir, 'validation')
#test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 1
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model_conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
train_features, train_labels = extract_features(train_dir, 10)
#validation_features, validation_labels = extract_features(validation_dir, 1000)
#test_features, test_labels = extract_features(test_dir, 1000)

print(train_features.shape)
print(train_labels.shape)

#print(train_features)
print(train_labels)
