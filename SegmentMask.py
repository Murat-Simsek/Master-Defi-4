#!/usr/bin/env python
# coding: utf-8

# 
# # Deep Learning example: U-Net for Phase Contrast segmentation
# 
# ## Introduction
# 
# This is a notebook that shows how to design and train a U-Net-like network for Miscroscopy Phase Contrast images.
# 
# The aim is to train the network using Phase contrast images as input, and the bacteria contours or masks (example below) as output.
# 
# 
# ## Data
# 
# Strain 4088 from JG @ CJW lab - segmentation was obtained from Oufti
# 
# Code adapted from
# Authors: Ignacio Arganda-Carreras, Andoni Rodriguez.
# Training School # 4 of NEUBIAS COST Action February 29th-March 3rd, 2020, Bordeaux

# In[1]:


import keras
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
import skimage
import skimage.io as io
import matplotlib.pyplot as plt


# In[2]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


import os
cwd = os.getcwd()
print(cwd)


# In[4]:


import os

# Path to the training images
train_path = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/train_img/0/'
#train_path_1 = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/train_images/1/'
# Read the list of file names
train_filenames = sorted([x for x in os.listdir( train_path ) if x.endswith(".png")])
#train_filenames_1 = sorted([x for x in os.listdir( train_path_1 ) if x.endswith(".png")])
#train_filenames = train_filenames + train_filenames_1
#train_filenames = sorted(train_filenames)

print( 'Images loaded: ' + str( len(train_filenames)) )

#for x in train_filenames:
#  print(x)


# In[5]:


from skimage.util import img_as_ubyte
from skimage import io
from matplotlib import pyplot as plt
from skimage import filters

# read training images
train_img = [ img_as_ubyte( io.imread( train_path + x ,as_gray=True) ) for x in train_filenames ]
#train_img_1 = [ img_as_ubyte( io.imread( train_path_1 + x ) ) for x in train_filenames_1 ]
#train_img = train_img + train_img_1



# In[6]:


# Path to the training images
test_path2 = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/mask_image_train_1/'


# Read the list of file names
test_filenames2 = sorted([x for x in os.listdir( test_path2 ) if x.endswith(".png")])

print( 'Images loaded: ' + str( len(test_filenames2)) )
#for x in train_filenames2:
#  print(x)

# read training images
test_img = [ img_as_ubyte( io.imread( test_path2 + x) ) for x in test_filenames2 ]





# In[7]:



plt.figure(figsize=(15,5))

# display one image of the train images
plt.subplot(1, 2, 1)
plt.imshow( train_img[250], 'gray' )
plt.title( 'Raw Phase Contrast Image' )

# display one image of the test images
plt.subplot(1, 2, 2)
plt.imshow( test_img[250], 'gray' )
plt.title( 'Segmented Image' )


# Create the training set by cropping the input images into patches of 512 x 512 pixels from original images 

# In[8]:


# create patches of 512x512 pixels => split each image in 2x2 tiles

def create_patches( imgs, num_x_patches, num_y_patches ):
    ''' Create a list of images patches out of a list of images
    Args:
        imgs: list of input images
        num_x_patches: number of patches in the X axis
        num_y_patches: number of patches in the Y axis
        
    Returns:
        list of image patches
    '''
    original_size = imgs[0].shape
    patch_width = original_size[ 0 ] // num_x_patches
    patch_height = original_size[ 1 ] // num_y_patches
    
    patches = []
    for n in range( 0, len( imgs ) ):
        image = imgs[ n ]
        for i in range( 0, num_x_patches ):
            for j in range( 0, num_y_patches ):
                patches.append( image[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ] )
    return patches

# use method to create patches
# initial image of 
train_patches = create_patches( train_img, 4, 4 )
test_patches = create_patches(test_img, 4, 4 )


plt.figure(figsize=(15,10))

# display one patch
plt.subplot(2, 3, 1)
plt.imshow( train_patches[5], 'gray', vmin = 0, vmax = 255)
plt.title( 'Training patch 5' )

# display one patch
plt.subplot(2, 3, 2)
plt.imshow( train_patches[9], 'gray' , vmin = 0, vmax = 255)
plt.title( 'Training patch 9' )
# display one patch
plt.subplot(2, 3, 3)
plt.imshow( train_patches[23], 'gray', vmin = 0, vmax = 255)
plt.title( 'Training patch 23' )

# display one patch
plt.subplot(2, 3, 4)
plt.imshow( test_patches[5], 'gray', vmin = 0, vmax = 255)
# display one patch
plt.subplot(2, 3, 5)
plt.imshow( test_patches[9], 'gray' , vmin = 0, vmax = 255)
# display one patch
plt.subplot(2, 3, 6)
plt.imshow( test_patches[23], 'gray' , vmin = 0, vmax = 255)


# We will use these patches as "ground truth" for training


# ## Network definition
# Next, we define our U-Net-like network, with 3 resolution levels in the contracting path, a bottleneck, and 3 resolution levels in the expanding path:
# 
# <figure>
# <center>
# <img src="https://drive.google.com/uc?id=1kjjBP4bTmDUKknRyEn1sF-sG1YZXmSxJ" width="750">
# </figure>
# 
# 
# 
# As loss function, we use the mean squared error (MSE) between the expected and the predicted pixel values, and we also include the mean absolute error (MAE) as a control metric.
# 

# In[9]:


# Input image size
print("Number of patches : ", len(train_patches))
patch_shape = train_patches[0].shape
seg_shape = test_patches[0].shape

train_width = patch_shape[0]
seg_width = seg_shape[0]
print("Train width :", train_width)
print("Seg width :", seg_width)

train_height = patch_shape[1]
seg_height = seg_shape[1]
print("Train height :", train_height)
print("Seg height :", seg_height)


# In[10]:


# Create U-Net for super-resolution

from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers import Dropout
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D
from keras.layers import concatenate

train_width = 224
train_height = 224
inputs = Input((train_width, train_height, 1))

c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
p1 = AveragePooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = AveragePooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = AveragePooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c4)
u5 = concatenate([u5, c3])
c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u5)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c2])
c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.1) (c6)
c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c1], axis=3)
c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.1) (c7)
c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

model = Model(inputs=[inputs], outputs=[outputs])
# compile the model with RMSProp as optimizer, MSE as loss function and MAE as metric
model.compile(optimizer= 'rmsprop', loss="binary_crossentropy")
model.summary()


# ## Training the network
# To follow Tensorflow standards, the input and output of the network have to be reshaped ot 256 x 256 x 1. Notice both input and ground truth images have their intensities scaled between 0.0 and 1.0.
# 
# Important training information:
# *   `Validation split`: percentage of training samples used for validation. Set to a random 10%.
# *   `Epochs`: which defines the maximum number of epochs the model will be trained. Initially set to 20.
# *   `Patience`: number of epochs that produced the monitored quantity (validation MSE) with no improvement after which training will be stopped. Initially set to 5.
# *   `Batch size`:  the number of training examples in one forward/backward pass. Initially set to 6.
# 

# In[11]:


# Train the network
from keras.callbacks import EarlyStopping
import numpy as np
import cv2
numEpochs = 10
earlystopper = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

# training input
input_shape = ( train_width, train_height, 1 ) # 512x512x1
# maxVal = np.amax(test_patches) *1.1
# minVal = np.amin(test_patches) * 0.75
# X_train = [(x-minVal)/(maxVal-minVal) for x in train_patches] # normalize between 0 and 1
X_train = [cv2.resize(x, (224, 224)) for x in train_img]
X_train = [x / 2**7 for x in X_train] # normalize between 0 and 1
X_train = [(1-x) for x in X_train] # normalize between 0 and 1
X_train = [np.reshape(x, input_shape ) for x in X_train]
X_train = np.asarray(X_train)


# In[12]:


# training ground truth
output_shape = ( train_width, train_height, 1 ) # 512x512x1

Y_train = [x/255  for x in test_img] # normalize between 0 and 1
Y_train = [(1-x) for x in Y_train]
Y_train = [np.reshape(x, output_shape ) for x in Y_train]

Y_train = np.asarray(Y_train)


# In[13]:


plt.figure(figsize=(10,5))

# display one patch
plt.subplot(1, 2, 1)
plt.imshow( X_train[250].reshape(224,224), 'gray', vmin=0, vmax=1)
plt.title( 'Training patch 5 x' )
plt.subplot(1, 2, 2)
plt.imshow( Y_train[250].reshape(224,224), 'gray', vmin=0, vmax=1)
plt.title( 'Training patch 5 y' )


# In[14]:


# Train the model using a 10% validation split and batch size of 6
history = model.fit( X_train, Y_train, validation_split=0.1, batch_size = 6,
                    epochs=numEpochs, callbacks=[earlystopper])


# In[37]:

import keras
import os
model = keras.models.load_model("my_h5_model_10_e.h5")
model = keras.models.load_model('C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/my_h5_model_10_e.h5.h5')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

# summarize history for loss
#plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

# # summarize history for MAE
# plt.subplot(1, 2, 2)
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('model MAE')
# plt.ylabel('MAE')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()


# In[ ]:





# ## Lire les images tests

# In[38]:


# Now we load some unseen images for testing
test_path = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/test_img/0/'

# Read the list of file names
test_filenames = sorted([x for x in os.listdir( test_path ) if x.endswith(".png")])

print( 'Images loaded: ' + str( len(test_filenames)) )

# Read test images
test_imgx = [ img_as_ubyte( io.imread( test_path + x,as_gray=True ) ) for x in test_filenames ]

# Create patches the same way as before
#test_patches = create_patches( test_imgx, 4,4 )

output_shape = ( train_width, train_height, 1 ) # 512x512x1

# maxVal = np.amax(test_patches) *1.1
# minVal = np.amin(test_patches) * 0.75
# X_test = [(x-minVal)/(maxVal-minVal) for x in test_patches] # normalize between 0 and 1
X_test = [cv2.resize(x, (224, 224)) for x in test_imgx]
X_test = [x/ 2**7 for x in X_test] # normalize between 0 and 1
X_test = [(1-x) for x in X_test]
X_test = [np.reshape(x, output_shape ) for x in X_test]
X_test = np.asarray(X_test)


plt.figure(figsize=(10,5))

# display one patch
plt.subplot(1, 3, 1)
plt.imshow( X_test[5,:,:,0], 'gray', vmin=0, vmax=1)
plt.subplot(1, 3, 2)
plt.imshow( X_test[15].reshape(512,512), 'gray', vmin=0, vmax=1)
plt.subplot(1, 3, 3)
plt.imshow( X_test[25].reshape(512,512), 'gray', vmin=0, vmax=1)


# In[17]:


np.amax(test_patches)


# In[18]:


#Now we load some unseen ref for testing
test_path2 = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/mask_image_test_1/'

test_filenames2 = sorted([x for x in os.listdir( test_path2 ) if x.endswith(".png")])

print( 'Available test images: ' + str( len(test_filenames2)) )

# Read test images
test_imgy = [ img_as_ubyte( io.imread( test_path2 + x ) ) for x in test_filenames2 ]

# Create patches the same way as before
# testseg_patches = create_patches( test_imgy,4,4)

output_shape = ( train_width, train_height, 1 ) # 512x512x1

Y_test = [x/255  for x in test_imgy] # normalize between 0 and 1
Y_test = [(1-x) for x in Y_test]
Y_test = [np.reshape(x, output_shape ) for x in Y_test]

Y_test = np.asarray(Y_test)

plt.figure(figsize=(10,5))

# display one patch
plt.subplot(1, 3, 1)
plt.imshow(Y_test[5,:,:,0], 'gray', vmin=0, vmax=1)
plt.subplot(1, 3, 2)
plt.imshow(Y_test[15,:,:,0], 'gray', vmin=0, vmax=1)
plt.subplot(1, 3, 3)
plt.imshow( Y_test[25].reshape(512,512), 'gray', vmin=0, vmax=1)


# In[19]:


print('\n# Generate predictions for x samples')
predictions = model.predict(X_test)
print('predictions shape:', predictions.shape)


# In[20]:


X_train[1:2,:,:,].shape
Y_test.shape


# In[21]:


from skimage.util import invert
plt.figure(figsize=(20,15))

img = 135
plt.subplot(3, 4, 1)
plt.imshow( invert(X_test[img,:,:,0]), 'gray' )
plt.title( 'Phase Contrast' )


plt.subplot(3, 4, 2)
plt.imshow( invert(predictions[img,:,:,0]), 'gray' )
plt.title( 'Predicted' )

plt.subplot(3, 4, 3)
plt.imshow( invert(Y_test[img,:,:,0]), 'gray' )
plt.title( 'Ground truth' )

plt.subplot(3, 4, 4)
plt.imshow( predictions[img,:,:,0] - Y_test[img,:,:,0], 'jet' )
plt.title( 'Difference Pred-GT' )

img = 181

plt.subplot(3, 4, 5)
plt.imshow( invert(X_test[img,:,:,0]), 'gray' )
plt.title( 'Phase Contrast' )


plt.subplot(3, 4, 6)
plt.imshow( invert(predictions[img,:,:,0]), 'gray' )
plt.title( 'Predicted' )

plt.subplot(3, 4, 7)
plt.imshow( invert(Y_test[img,:,:,0]), 'gray' )
plt.title( 'Ground truth' )

plt.subplot(3, 4, 8)
plt.imshow( predictions[img,:,:,0] - Y_test[img,:,:,0], 'jet' )
plt.title( 'Difference Pred-GT' )

img = 1015

plt.subplot(3, 4, 9)
plt.imshow( invert(X_test[img,:,:,0]), 'gray' )
plt.title( 'Phase Contrast' )


plt.subplot(3, 4, 10)
plt.imshow( invert(predictions[img,:,:,0]), 'gray' )
plt.title( 'Predicted' )

plt.subplot(3, 4, 11)
plt.imshow( invert(Y_test[img,:,:,0]), 'gray',vmin=0,vmax=1)
plt.title( 'Ground truth' )

plt.subplot(3, 4, 12)
plt.imshow( predictions[img,:,:,0] - Y_test[img,:,:,0], 'jet' )
plt.title( 'Difference Pred-GT' )


x=130
    
# In[22]:


plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
plt.imshow( invert(predictions[50,180:280,155:255,0]), 'jet' )
plt.title( 'Predicted example 1' )
plt.subplot(1, 3, 2)
plt.imshow( invert(predictions[50,90:190,260:360,0]), 'jet' )
plt.title( 'Predicted example 2' )
plt.subplot(1, 3, 3)
plt.imshow( invert(Y_test[50,90:190,260:360,0]), 'jet' )
plt.title( 'GT example 2' )


# In[23]:


os.chdir('C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/')
# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
model.save("my_h5_model_10_e.h5")


# In[24]:


plt.figure(figsize=(15,15))
for i in range(0,16):
    plt.subplot(4, 4, i+1)
    plt.imshow( invert(predictions[i,:,:,0]),'gray')
    plt.title(i)
    
predictions[0].shape
len(predictions)

imRes = skimage.util.montage(predictions[0:15,:,:,0])
imRes.shape


# In[25]:


def undo_patches( patches, num_x_patches, num_y_patches ):
    ''' Create a list of images out of a list of images patches (undo create_patches)
    Args:
        patches: list of input patches
        num_x_patches: number of patches in the X axis in the final image
        num_y_patches: number of patches in the Y axis in the final image
        
    Returns:
        list of images
    '''
    patch_size = patches[0].shape
    img_width = patch_size[ 0 ] * num_x_patches
    img_height = patch_size[ 1 ] * num_y_patches
    n_images = int(len(patches) / ( num_x_patches * num_y_patches))
    
    imgs = []
    for n in range( 0, n_images-1):
        nstart = n * 16
        nstop = n * 16 + 15
        #print('n = ' + str(n) + '; nstart = '+ str(nstart) + '; nstop = '+ str(nstop) )
        imgs.append(skimage.util.montage(patches[nstart:nstop,:,:,0]))
    return imgs


# In[26]:


segmented_images = undo_patches(predictions,4,4)


# In[27]:


segmented_images = [invert(x)  for x in segmented_images] 
plt.figure(figsize=(15,15))
plt.imshow( segmented_images[0], 'jet')


# In[28]:


def doItAll(imgAsegmenter, num_x_patches, num_y_patches, model ): 
    test_patches = create_patches( imgAsegmenter, 4,4 )
    output_shape = ( 512, 512, 1 ) # 512x512x1
#     maxVal = np.amax(imgAsegmenter) *1.1
#     minVal = np.amin(imgAsegmenter) * 0.75
#     X_test = [(x-minVal)/(maxVal-minVal)  for x in test_patches] # normalize between 0 and 1
    X_test = [x / 2**7  for x in test_patches] # normalize between 0 and 1
    X_test = [(1-x) for x in X_test]
    X_test = [np.reshape(x, output_shape ) for x in X_test]
    X_test = np.asarray(X_test)
    print('predicting...')
    predictions = model.predict(X_test)
    segmented_images = undo_patches(predictions,4,4)
    segmented_images = [invert(x)  for x in segmented_images] 
    print('done')
    return segmented_images
 
def doIt512(imgAsegmenter, num_x_patches, num_y_patches, model ): 
    #test_patches = create_patches( imgAsegmenter, 4,4 )
    output_shape = ( 512, 512, 1 ) # 512x512x1
#     maxVal = np.amax(imgAsegmenter) *1.1
#     minVal = np.amin(imgAsegmenter) * 0.75
#     X_test = [(x-minVal)/(maxVal-minVal)  for x in test_patches] # normalize between 0 and 1
    X_test = [x / 2**7  for x in test_patches] # normalize between 0 and 1
    X_test = [(1-x) for x in X_test]
    X_test = [np.reshape(x, output_shape ) for x in X_test]
    X_test = np.asarray(X_test)
    print('predicting...')
    predictions = model.predict(X_test)
    #segmented_images = undo_patches(predictions,4,4)
    segmented_images = [invert(x)  for x in predictions] 
    print('done')
    return segmented_images


# In[29]:


imgSeg = doItAll(test_imgx, 4, 4, model)


# In[30]:


print(""+ str(len(imgSeg)))
plt.figure(figsize=(15,15))
for i in range(0,4):
    plt.subplot(2, 2, i+1)
    plt.imshow( test_imgx[i],'gray')
    
plt.figure(figsize=(15,15))
for i in range(0,4):
    plt.subplot(2, 2, i+1)
    plt.imshow( imgSeg[i],'gray')


# In[31]:


#Now we load some unseen ref for testing
test_path3 = 'testPseudo/'

test_filenames3 = sorted([x for x in os.listdir(test_path3) if x.endswith(".tif")])

print( 'Available test images: ' + str( len(test_filenames3)) )

# Read test images
test_imgy3 = [ img_as_ubyte( io.imread( test_path3 + x ) ) for x in test_filenames3 ]


# In[32]:


imgSeg = doIt512(test_imgy3, 4, 4, model)


# In[33]:


print(""+ str(len(imgSeg)))
imgSeg[0].shape
    


# In[34]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow( test_imgy3[0], 'gray', vmin=50, vmax=300 )
plt.subplot(1, 2, 2)
plt.imshow( imgSeg[0][:,:,0], 'jet' )


# #### Ccl partielle
# 
# Ca fonctionne a peu près avec pseudo en forme rods
# Mieux si apprentissage correspondant

# In[35]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow( test_imgy3[1], 'gray', vmin=50, vmax=300 )
plt.subplot(1, 2, 2)
plt.imshow( imgSeg[1][:,:,0], 'gray' )


# ### CCl partielle
# 
# Ca marche pas quand la taille des bact est très différente

# In[36]:


#Now we load some unseen ref for testing
test_path3 = 'testPseudo2/'
test_filenames3 = sorted([x for x in os.listdir(test_path3) if x.endswith(".tif")])
print( 'Available test images: ' + str( len(test_filenames3)) )
# Read test images
test_imgy3 = [ img_as_ubyte( io.imread( test_path3 + x ) ) for x in test_filenames3 ]

imgSeg = doIt512(test_imgy3, 4, 4, model)

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow( test_imgy3[1], 'gray')
plt.subplot(1, 2, 2)
plt.imshow( imgSeg[1][:,:,0], 'jet' )


# In[ ]:




