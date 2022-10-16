DATA_URL = 'C://Users//pierluigi.sicuro//Desktop//ds//flower_photos//dandelion//7355522_b66e5d3078_m.jpg'

'''
import cv2
im = cv2.imread(DATA_URL,mode='RGB')


import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
    
'''    
    
'''    
from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

x = load_image(DATA_URL)

import matplotlib.pyplot as plt
imgplot = plt.imshow(x)
plt.show()
'''



import matplotlib.image as mpimg

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images    



