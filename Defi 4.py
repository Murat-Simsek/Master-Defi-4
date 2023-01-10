#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pydicom
import os
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


#pip install pydicom


# In[3]:


df = pd.read_csv('trainSet-rle.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df['EncodedPixels'] = np.where(df['EncodedPixels']==' -1', 0,1 )   


# In[7]:


df['ImageId'].nunique()


# In[8]:


PathDicom = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/dicom-images-train'

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
   


# In[9]:


len(lstFilesDCM)


# In[10]:


imgL = [pydicom.read_file(x) for x in lstFilesDCM]


# In[11]:


len(imgL)


# In[12]:


scans=[]
for file in lstFilesDCM:
          dicomds = pydicom.read_file(file,)
          scans.append(dicomds)


# In[13]:


scans[0]


# In[14]:


# df_img_dataset = pd.DataFrame(scans)


# In[15]:


# df_img_dataset[13][1][-1]


# In[ ]:





# In[ ]:





# In[16]:


# img_list_id = []
# for i in range(9712):
#     sop = imgL[i]['SOPInstanceUID'][:100]['PatientsSex']
#     img_list_id.append(sop)


# In[17]:


# imgL[0]['SOPInstanceUID']


# In[18]:


# img_list = pd.DataFrame(img_list_id,columns=['ImageId'])


# In[19]:


# imgL[0].to_dataframe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pydicom
import pandas as pd
import os

# Get a list of all the DICOM files in the directory
files = lstFilesDCM

# Create an empty list to store the dataframes
dataframes = []
i=0
# Loop through the files
for file in files:
    # Read the DICOM file
    dcm_data = pydicom.dcmread(file)

    # Extract the age and sex information from the DICOM file
    age = dcm_data.PatientAge
    sex = dcm_data.PatientSex
    ids = dcm_data.SOPInstanceUID

    # Create a dataframe with the age and sex information
    data = {"age": age, "sex": sex,"ImageId" : ids}
    df_s = pd.DataFrame(data,index=[i])
    i=i+1

    # Add the dataframe to the list of dataframes
    dataframes.append(df_s)

# Concatenate all the dataframes into a single dataframe
df_i = pd.concat(dataframes)


# In[ ]:


df_i


# In[ ]:


df


# In[ ]:


df_f = pd.merge(df,df_i,on="ImageId",how="right")


# In[ ]:


len(df_f['age'])


# In[ ]:


df_f


# In[ ]:


df_f.drop_duplicates("ImageId",inplace=True)


# In[ ]:


df_f.reset_index(drop=True,inplace=True)


# In[ ]:





# In[ ]:


# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler



# # Create a scaler object
# scaler = MinMaxScaler(feature_range=(0, 100))

# # Scale and encode the values in the "age" column
# df_f["encoded_age"] = df_f.apply(lambda x: (scaler.fit_transform(df_f.loc[x.index, "age"].values.reshape(-1, 1)).flatten()[0] + 128) if x["sex"] == "M" else scaler.fit_transform(df_f.loc[x.index, "age"].values.reshape(-1, 1)).flatten()[0], axis=1)


# In[ ]:


# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Create a scaler object
# scaler = MinMaxScaler(feature_range=(0, 100))

# # Scale and encode the values in the "age" column
# df_f["encoded_age"] = df_f.apply(lambda x: (scaler.fit_transform(df_f.loc[x.index, "age"].values.reshape(-1, 1)).flatten()[0] + 128) if x["sex"] == "M" else scaler.fit_transform(df_f.loc[x.index, "age"].values.reshape(-1, 1)).flatten()[0], axis=1)


# In[ ]:


def FemmesScale(age):
  # Vérifie que l'âge est bien compris entre 0 et 100
  if age < 0 or age > 120:
    raise ValueError("L'âge doit être compris entre 0 et 100")

  # Calcule l'âge recodé entre 0 et 132
  scaled_age = age * 127 / 100

  return scaled_age

def HommesScale(age):
  # Vérifie que l'âge est bien compris entre 0 et 100
  if age < 0 or age > 120:
    raise ValueError("L'âge doit être compris entre 0 et 100")

  # Calcule l'âge recodé entre 128 et 255
  scaled_age = age * (255 - 128) / 100 + 128

  return scaled_age


# In[ ]:


HommesScale(65)


# In[ ]:


df_f['age'] = df_f['age'].astype("int")
df_f['EncodedPixels'].fillna(0,inplace=True) # Replace NaN by 0 
df_f['sex'] = df_f['sex'].replace("F",0)
df_f['sex'] = df_f['sex'].replace("M",1)


# In[ ]:


df_f[df_f['age']>100]


# In[ ]:


def replace_age(age):
    if age > 100:
        return df_f['age'].mean()
    else:
        return age

df_f['age'] = df_f['age'].apply(replace_age)


# In[ ]:


df_f[df_f['EncodedPixels'].isnull()]


# In[ ]:


df_f.info()


# In[ ]:





# In[ ]:





# In[ ]:


# ls = []
# for x in df_f[['age','sex']:
#     if (df_f['sex'][x] == 0):
#         df_s = FemmesScale(df_f['age'][x])
#         ls.append(df_s)
#     else:
#         df_s = HommesScale(df_f['age'][x])
#         ls.append(df_s)
#         #print('hi')
# df_f['encoded_age'] = ls


# In[ ]:


#   df_f["scale"] = ""
  
#   # Iterate over the rows of the DataFrame
#   for index, row in df_f.iterrows():
#     # If the row is a male, apply the HommesScale function to the age column
#     if row["sex"] == "M":
#       df_f.at[index, "scale"] = HommesScale(row["age"])
#     # If the row is a female, apply the FemmesScale function to the age column
#     elif row["sex"] == "F":
#       df_f.at[index, "scale"] = FemmesScale(row["age"])


# In[ ]:



  df_f["encoded_age"] = ""
  
  # Define a dictionary with the mapping of sex to the appropriate function
  sex_to_function = {1: HommesScale, 0: FemmesScale}
  
  # Iterate over the rows of the DataFrame
  for index, row in df_f.iterrows():
    # Use the dictionary to lookup the appropriate function for the sex
    # and apply it to the age column
    df_f.at[index, "encoded_age"] = sex_to_function[row["sex"]](row["age"])


# In[ ]:


df_f['encoded_age']=df_f['encoded_age'].astype("int")
df_f['EncodedPixels']=df_f['EncodedPixels'].astype('int')


# In[ ]:


# # df_f["encoded_age"] = df_f['age'].apply(lambda x: HommesScale(x) if (df_f['sex'] == "F") else FemmesScale(x))
# df_f["encoded_age"] = df_f.apply(lambda x: HommesScale(x.age) if (x['sex'] == "F") else FemmesScale(x.age))
# df_f['encoded_age']=df_f['encoded_age'].astype('int')
# df_f    


# In[ ]:


df_f


# In[ ]:


#img = img_list.merge(df,how='left',on='ImageId')


# In[ ]:


#img.info()


# In[ ]:


#df['compare'] = (df['ImageId'] == df['img_list'])


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from keras.utils import np_utils


# In[ ]:


#ds = pydicom.read_file(lstFilesDCM[1])
ds = pydicom.dcmread(lstFilesDCM[1])
data = ds.pixel_array
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
data.shape
from skimage.transform import resize
IMG_PX_SIZE = 256
resized_img = resize(data, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
resized_img.shape
plt.imshow(resized_img , cmap="gray", vmin=0, vmax=255)


# In[ ]:


ds = pydicom.dcmread(lstFilesDCM[1])
data = ds.pixel_array
data.shape
plt.imshow(data , cmap="gray", vmin=0, vmax=255)
data.shape


# In[ ]:





# In[ ]:





# In[ ]:


#pip install opencv-python


# In[ ]:


# Resize images
import cv2
import pydicom
import os
import shutil
from skimage.transform import resize

# List of image file paths
image_paths = lstFilesDCM

# Create the new folder
new_folder = 'resized_images'
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

resized_img = []
# Loop through the list of images
for image_path in image_paths:
    # Load the DICOM image using pydicom
    ds = pydicom.dcmread(image_path)
#     # Convert the image data to a NumPy array
#     image_data = ds.pixel_array
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

#     # Resize the image using OpenCV's resize function
#     resized_image = resize(image_data, (512, 512), anti_aliasing=True)
#     #resized_img.append(resized_image)
    
    data = ds.pixel_array
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data.shape
    from skimage.transform import resize
    IMG_PX_SIZE = 128
    resized_img = cv2.resize(data, (IMG_PX_SIZE, IMG_PX_SIZE))
    
    

    # Get the base file name
    base_name = os.path.basename(image_path)
    #pydicom.dcmwrite(os.path.join(new_folder, base_name), resized_image)


    # Save the resized image as a PNG file
    cv2.imwrite(os.path.join(new_folder, base_name + '.png'), resized_img)
    #cv2.imwrite(os.path.join(new_folder, base_name), resized_image)


# In[ ]:





# In[ ]:


ar_imgL=[]
for x in imgL:
    y = x.pixel_array
    ar_imgL.append(y)


# In[ ]:


ar_imgL = np.array(ar_imgL)


# In[ ]:


ar_imgL[0]


# In[ ]:


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(ar_imgL, df_f["EncodedPixels"], test_size=0.2)


# In[70]:





# In[73]:


type(X_train)


# In[74]:


ar_imgL.shape


# In[75]:


type(ar_imgL)


# In[76]:


print(X_train.shape)
print(X_test.shape)


# In[92]:


# import cv2
# import pydicom
# import os

# # List of image file paths
# image_paths = lstFilesDCM

# # Create the new folder
# new_folder = 'resized_images'
# if not os.path.exists(new_folder):
#   os.makedirs(new_folder)

# # Loop through the list of images
# for image_path in image_paths:
#   # Load the DICOM image using pydicom
#       ds = pydicom.dcmread(image_path)
# # Convert the image data to a NumPy array
#       image_data = ds.pixel_array
#       #image_data = image_data.astype('uint8')  # or any other compatible data type

#       # Resize the image using OpenCV's resize function
#       resized_image = cv2.resize(image_data, (256, 256))

#       # Encapsulate the resized image data
#       encapsulated_image_data = pydicom.encaps.encapsulate(resized_image)

#       # Update the DICOM image data with the encapsulated image data
#       ds.PixelData = encapsulated_image_data

#       # Update the image dimensions in the DICOM metadata
#       ds.Rows, ds.Columns = 256, 256

#       # Get the base file name
#       base_name = os.path.basename(image_path)

#       # Save the resized image to the new folder
#       ds.save_as(os.path.join(new_folder, base_name))


# In[ ]:


# import cv2
# import pydicom
# import os

# # List of image file paths
# image_paths = lstFilesDCM

# # Create the new folder
# new_folder = 'resized_images'
# if not os.path.exists(new_folder):
#   os.makedirs(new_folder)

# # Loop through the list of images
# for image_path in image_paths:
#   # Load the DICOM image using pydicom
#     ds = pydicom.dcmread(image_path)

#       # Convert the image data to a NumPy array
#     image_data = ds.pixel_array
#     from skimage.transform import resize
#     IMG_PX_SIZE = 251

#     resized_img = resize(image_data, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)

#       # # Resize the image using OpenCV's resize function
#       # resized_image = cv2.resize(image_data, (256, 256))
#       # # Get the base file name
#     base_name = os.path.basename(image_path)
#       # # Save the resized image to the new folder
#     ds.PixelData = resized_img.tobytes()
#     ds.save_as(os.path.join(new_folder, base_name))


# In[49]:


PathDicom = 'C:/Users/Casper/Desktop/Master1-Cours/rstudio and Python/Application pratique Defi 4/resized_images'

lstFilesDCM_resized = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_resized.append(os.path.join(dirName,filename))


# In[50]:


len(lstFilesDCM_resized)


# In[47]:


# # Convert the DICOM image to a PIL image
# pil_image = Image.fromarray(images.pixel_array)
# # Resize the image to a smaller size
# new_size = (512, 512)
# resized_image = pil_image.resize(new_size)
# resized_array = np.array(resized_image)


# In[51]:


from PIL import Image

dicom_images = []
for dicom_file in lstFilesDCM_resized:
    #images = plt.imread(dicom_file)
    #images = imageio.imread(dicom_file,)
    #images = Image.open(dicom_file).convert('L')
    images = Image.open(dicom_file)

    images = np.array(images)

    dicom_images.append(images)
    

# Extract the pixel data from the DICOM images and convert it to a NumPy array
# image_data = [image.pixel_array for image in dicom_images]
# image_data = np.array(image_data)



# In[52]:


type(dicom_images[0])


# In[53]:


dicom_images[0].shape


# In[54]:


dicom_images[0]


# In[55]:


dicom_images = np.array(dicom_images)


# In[62]:


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(dicom_images, df_f["EncodedPixels"], test_size=0.2)


# In[63]:


dicom_images.shape


# In[64]:


print(X_train.shape)
print(X_test.shape)


# In[65]:


# dim = 128*128*3
# X_train = X_train.reshape(7769, dim)
# X_test = X_test.reshape(1943, dim)


# In[107]:


Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)


# In[67]:


X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255


# In[ ]:


# y_test = y_test.astype('float16')
# y_train = y_train.astype('float16')


# In[ ]:


#X_traing_in = []
# for i in imgL:
#     x = imgL.pixel_array
#     img_in.append(x)
# img_in = np.array(img_in)       


# #### Model 

# In[68]:


from tensorflow.keras import layers, models
import tensorflow as tf
from keras.layers import Flatten, Input, concatenate


# In[183]:


dim = 128
model = Sequential()
model.add(Dense(dim, input_shape=(dim,128,3)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(216))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam')


# In[78]:


model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(layers.Dense(10))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()


# In[79]:


model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics='accuracy')


# In[80]:


Y_test[5]


# In[81]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])


# In[82]:


history = model.fit(X_train, Y_train, epochs=1, 
                    validation_data=(X_test, Y_test),verbose=1)


# In[84]:


model.evaluate(X_test,Y_test)


# In[100]:


from sklearn.metrics import classification_report

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(y_pred_bool)
print(classification_report(Y_test, y_pred_bool))


# In[99]:


pred[100]


# In[88]:


pred = model.predict(X_test)
result = np.where(pred > 0.5, 1, 0) #<--to get the binary category
print(result)


# In[106]:


len(result[result == [0,1]])


# In[89]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)


# In[ ]:





# In[260]:


nb_epochs = 1
#TODO ajouter un vrai split sur train
model.fit(X_train, y_train,
          batch_size=32, epochs=nb_epochs,
          validation_data=(X_test, y_test))


# In[298]:


predi = model.predict(X_test)


# In[300]:


predi[0:50]


# In[188]:


pred = (model.predict(X_test) > 0.2).astype("int64")


# In[189]:


from sklearn.metrics import classification_report,confusion_matrix


# In[190]:


pred[]


# In[ ]:





# In[102]:


print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


# In[ ]:





# ### Add borders

# In[67]:


for i,x in zip(lstFilesDCM[0:4],df_f['encoded_age']):
    imgL = pydicom.read_file(i)
    image = imgL.pixel_array
    image = np.pad(image, ((100, 100), (100, 100)), mode="constant", constant_values=x)
    plt.imshow(image , cmap="gray", vmin=0, vmax=255)
    print(x)
    


# In[ ]:





# In[55]:


plt.figure()
image = imgL[1].pixel_array
imag = imgL[1].pixel_array
# Add white pixels around the image
image = np.pad(image, ((100, 100), (100, 100)), mode="constant", constant_values=155)
f,ax = plt.subplots(1,2)
ax[0].imshow(image , cmap="gray", vmin=0, vmax=255)
ax[1].imshow(imag , cmap="gray", vmin=0, vmax=255)
# Save the padded image to a new DICOM file
#imgL[0].PixelData = image.tobytes()


# In[ ]:


image = imgL[1].pixel_array

# Add white pixels around the image
# image = np.pad(image, ((100, 100), (100, 100)), mode="constant", constant_values=155)


# ### MASK to RLE Function

# In[ ]:


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# In[ ]:


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(dicom_images, df_f[["EncodedPixels"]], test_size=0.2)


# In[99]:


from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam


# In[108]:


dicom_images.shape


# In[109]:


df_f


# In[168]:


import numpy as geek

xtrain = dicom_images[0:8000]
#xtrain = geek.squeeze(x_train)
xtest = dicom_images[8001:9712]
#xtest = geek.squeeze(x_test)

print("x_train shape:", xtrain.shape)
print(xtrain.shape[0], "train samples")
print(xtest.shape[0], "test samples")

df_train = df_f[['age','sex']][0:8000]
df_test = df_f[['age','sex']][8001:9712]


y_train = df_f['EncodedPixels'][0:8000]
y_test = df_f['EncodedPixels'][8001:9712]


# In[169]:


xtrain.shape


# In[162]:


import numpy as geek
out_arr = geek.squeeze(xtest) 


# In[163]:


out_arr.shape


# In[174]:


def create_cnn(width, height, depth, filters=(32, 64), regularizer=None):
    """
    Creates a CNN.
    """
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1
 
    # Define the model input
    inputs = Input(shape=inputShape)
 
    # Loop over the number of filters 
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
 
        # Create loops of CONV => RELU => BN => POOL layers
        x = Conv2D(f, kernel_size=(3, 3))(x)
        x = Activation("relu")(x)
        #x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x = Flatten()(x)
    #x = Dense(16, kernel_regularizer=regularizer)(x)
    #x = Activation("relu")(x)
    #x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
 
    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    #x = Dense(4, kernel_regularizer=regularizer)(x)
    #x = Activation("relu")(x)
 
    # Construct the CNN
    model = Model(inputs, x)
 
    # Return the CNN
    return model   


cnn = create_cnn(128, 128, 3)


# In[175]:


#######
# ffn for dataframe
###

def create_ffn(dim, regularizer=None):
    """Creates a simple two-layer MLP with inputs of the given dimension"""
    model = Sequential()
    model.add(Dense(7, input_dim=dim, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(4, activation="relu", kernel_regularizer=regularizer))
    return model

from sklearn.preprocessing import MinMaxScaler

cs = MinMaxScaler()
trainXdtf = cs.fit_transform(df_train)
testXdtf = cs.transform(df_test)


ffn = create_ffn(trainXdtf.shape[1])


# In[176]:



#######
# combine mixed data
###

combinedInput = concatenate([ffn.output, cnn.output])


x = Dense(1, activation="sigmoid")(combinedInput) #add an output layer

model1 = Model(inputs=[ffn.input, cnn.input], outputs=x)
model1.summary()
#dot_img_file = '/home/jgodet/Seafile/MaBibliotheque/Enseignements/Ens2022-23/IDS_Apps/defi4//model_1.png'
#tf.keras.utils.plot_model(model1, to_file=dot_img_file, show_shapes=True)


opt = tf.keras.optimizers.legacy.Adam()
model1.compile(loss="categorical_crossentropy", metrics=['acc'], optimizer=opt)


# In[182]:


model1_history = model1.fit(
  [trainXdtf,xtrain],
  y_train, 
  validation_split=0.1, 
  epochs=5, 
  batch_size=128)


# In[181]:


from sklearn.metrics import classification_report

y_pred = model.predict(xtest, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool,zero_division=0))


# In[ ]:




