#C:\Users\pierluigi.sicuro>cd C:\Users\pierluigi.sicuro\Desktop\Python\scr\my1 & venv\Scripts\activate
#(venv) C:\Users\pierluigi.sicuro\Desktop\Python\scr\my1>python C:\Users\pierluigi.sicuro\Desktop\Python\TensorFlow\mytfscr\vgg16\mytfscr-vgg16-my-heatmap.py
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model_conv_base = VGG16(weights='imagenet'
#, include_top=False
)

model_conv_base.summary()



img_path = 'C://Users//pierluigi.sicuro//Desktop//ds//newcases//134409839_71069a95d1_m.jpg'                                                                
#img_path = 'C://Users//pierluigi.sicuro//Desktop//ds//newcases//15987457_49dc11bf4b.jpg'   
#img_path = 'C://Users//pierluigi.sicuro//Desktop//ds//newcases//138166590_47c6cb9dd0.jpg'   

#img_path = 'C://Users//pierluigi.sicuro//Desktop//ds//flower_photos//roses//2951375433_ae2726d9d2_m.jpg'
#img_path = 'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-signed-test//SIAED202008201029068507.tif-11.png'
image_size = 224
img = image.load_img(img_path, target_size=(image_size, image_size))
#img = np.random.random((image_size, image_size, 3))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)



features = model_conv_base.predict(img_tensor)
max_feature = features[:, np.argmax(features[0])]
print('max_feature:', max_feature)
print('features.shape:', features.shape)
decoded_features = decode_predictions(features, top=3)[0]
decoded_features_array = np.array(decoded_features)
print('decoded_features_array.shape:', decoded_features_array.shape)
print('Predicted:', decoded_features)


# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
from tensorflow.keras import models
conv_layer = model_conv_base.get_layer("block5_conv3")
heatmap_model = models.Model([model_conv_base.inputs], [conv_layer.output, model_conv_base.output])

import tensorflow as tf
from tensorflow.keras import backend as K

with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(img_tensor)
    #conv_output, predictions = models.Model([model_conv_base.inputs], [model_conv_base.get_layer("block5_conv3").output, model_conv_base.output])(img_tensor)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

# Channel-wise mean of resulting feature-map is the heatmap of class activation
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat

print(heatmap.shape)

import matplotlib.pyplot as plt
# Render heatmap via pyplot
plt.matshow(heatmap[0])
plt.show()