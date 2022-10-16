'''
import PIL.Image
class Stack_wrapper(object):
    def __init__(self,fname):
        #fname is the full path
        self.im  = PIL.Image.open(fname)

        self.im.seek(0)
        # get image dimensions from the meta data the order is flipped
        # due to row major v col major ordering in tiffs and numpy
        self.im_sz = [self.im.tag[0x101][0],
                      self.im.tag[0x100][0]]
        self.cur = self.im.tell()

    def get_frame(self,j):
        #Extracts the jth frame from the image sequence. 
        #if the frame does not exist return None
        try:
            self.im.seek(j)
        except EOFError:
            return None

        self.cur = self.im.tell()
        return np.reshape(self.im.getdata(),self.im_sz)
    def __iter__(self):
        self.im.seek(0)
        self.old = self.cur
        self.cur = self.im.tell()
        return self

    def next(self):
        try:
            self.im.seek(self.cur)
            self.cur = self.im.tell()+1
        except EOFError:
            self.im.seek(self.old)
            self.cur = self.im.tell()
            raise StopIteration
        return np.reshape(self.im.getdata(),self.im_sz)
'''
'''
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA//SIAED202008201029065451.tif')
print(img.n_frames)

for i in range(img.n_frames):
    try:
        img.seek(i)
        framenumber=i+1
        print(framenumber)
        #print(img.getpixel( (0, 0)))
        #plt.figure(figsize=(10, 10))
        #plt.imshow(img)
        #plt.show()
        if((framenumber == 5)or(framenumber==7)or(framenumber==img.n_frames)):
            img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-signed//SIAED202008201029065451-' + str(framenumber) + '.png')
        else:
            img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-unsigned//SIAED202008201029065451-' + str(framenumber) + '.png')

    except EOFError:
        # Not enough frames in img
        break
'''

'''
import os
from PIL import Image
dirname='C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA'
filenames = os.listdir(dirname)
#print(filename)
for x in range(len(filenames)): 
    print(filenames[x])
    img = Image.open(dirname + '//' + filenames[x])
    print(img.n_frames)
    for i in range(img.n_frames):
        try:
            img.seek(i)
            framenumber=i+1
            print(framenumber)
            #print(img.getpixel( (0, 0)))
            #plt.figure(figsize=(10, 10))
            #plt.imshow(img)
            #plt.show()
            if(x<5):
                if((framenumber == 5)or(framenumber==7)or(framenumber==img.n_frames)):
                    img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-signed-test//' + filenames[x] + '-' + str(framenumber) + '.png')
                else:
                    img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-unsigned-test//' + filenames[x] + '-' + str(framenumber) + '.png')
            else:
                if((framenumber == 5)or(framenumber==7)or(framenumber==img.n_frames)):
                    img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-signed-training//' + filenames[x] + '-' + str(framenumber) + '.png')
                else:
                    img.save(r'C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA-unsigned-training//' + filenames[x] + '-' + str(framenumber) + '.png')                
                        
        except EOFError:
            # Not enough frames in img
            break       
'''

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('C://Users//pierluigi.sicuro//Desktop//ds//BG_QUA//SIAED202008201029065451.tif')
print(img.n_frames)

img.seek(0)
pix = img.load()
print(img.size[0])
print(img.size[1])
plt.figure(figsize=(10, 10))
plt.imshow(img)
#plt.show()

'''
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if(img.getpixel( (i, j))!=255): 
            print(img.getpixel( (i, j)))
'''
'''
sequence_of_pixels = img.getdata()
list_of_pixels = list(sequence_of_pixels)
print(list_of_pixels)
'''
crop_rectangle = (500, 500, 2000, 2000)
cropped_img = img.crop(crop_rectangle)
#cropped_img.show()

sequence_of_pixels = cropped_img.getdata()
list_of_pixels = list(sequence_of_pixels)
print(list_of_pixels)

