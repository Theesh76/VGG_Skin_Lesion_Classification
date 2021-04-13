# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:21:33 2021

@author: skris
"""

#%%
# Importing Required Libraries

import cv2                
import numpy as np             
import glob    
import json      
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import load_model
# %% Data Visualization   
with open('train_labels.json') as f:
    data = json.load(f)        
data = list(data.items())
labels = np.array(data)
y = labels[:,1]
labels_unique = np.unique(y, axis=0)  
print("The Lables that needs to be classified:",labels_unique)
with open('train_labels.json') as f:
    data = json.load(f)        
data = list(data.items())
labels = np.array(data)
y = labels[:,1]
labels, counts = np.unique(y,return_counts=True) 
ticks = range(len(counts))

print("The training set comprises of:")
for i in range(len(counts)):
    if labels[i] == 'bcc':
        print("Basal Cell Carcinoma (bcc) -",counts[i],"images")
    elif labels[i] == 'akiec':
        print("Actinic Keratoses and IntraEpithelial Carcinoma (akiec) -",counts[i],"images")
    elif labels[i] == 'df':
        print("DermatoFibroma (df) -",counts[i],"images")
    elif labels[i] == 'mel':
        print("MELanoma (mel) -",counts[i],"images")
    elif labels[i] == 'nv':
        print("melanocytic NeVi (nv) -",counts[i],"images")
    else:
        print("VASCular lesions (vasc) -",counts[i],"images")
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, labels)
plt.title('Data Visulaization of Training Set')
plt.xlabel('Different types of Skin Lesions')
plt.ylabel('Frequency of Skin Lesions')

# %%
# Data Preparation
images = []
image_paths = glob.glob( 'train/*.jpg' )
i=0
k1=k2=k3=k4=k5=k6=k7=0
for imagefile in image_paths:
    image = cv2.imread(imagefile)
    if y[i] == 'bcc':
        loc = 'Data_Analysis/bcc/' + str(k1) + '.jpg'
        cv2.imwrite(loc,image)
        k1 = k1 + 1
    elif y[i] == 'akiec':
        loc = 'Data_Analysis/akiec/' + str(k2) + '.jpg'
        cv2.imwrite(loc,image)
        k2 = k2 + 1
    elif y[i] == 'df':
        loc = 'Data_Analysis/df/' + str(k3) + '.jpg'
        cv2.imwrite(loc,image)
        k3 = k3 + 1
    elif y[i] == 'mel':
        loc = 'Data_Analysis/mel/' + str(k4) + '.jpg'
        cv2.imwrite(loc,image)
        k4 = k4 + 1
    elif y[i] == 'nv':
        loc = 'Data_Analysis/nv/' + str(k5) + '.jpg'
        cv2.imwrite(loc,image)
        k5 = k5 + 1
    elif y[i] == 'bkl':
        loc = 'Data_Analysis/bkl/' + str(k6) + '.jpg'
        cv2.imwrite(loc,image)
        k6 = k6 + 1
    else:
        loc = 'Data_Analysis/vasc/' + str(k7) + '.jpg'
        cv2.imwrite(loc,image)
        k7 = k7 + 1
    i = i+1    
# %% Data Augmentation

datagen = ImageDataGenerator(rotation_range=10, 
                               width_shift_range=0.05, 
                               height_shift_range=0.05, 
                               rescale=1/255,  
                               zoom_range=0.1, 
                               horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='nearest')
CATEGORIES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
label = np.zeros((1,7))
for category in CATEGORIES:
    image_paths = glob.glob( 'Data_Analysis/'+category+'/*.jpg' )  
    for imagefile in image_paths:
        image = cv2.imread(imagefile)
        x = np.asarray(image)
        x = x.reshape((1,) + x.shape)  
        i = 0
        for batch in datagen.flow(x, batch_size=1,save_to_dir='Data_Analysis/'+category, save_prefix='aug', save_format='jpg'):
            if category=='akiec' and i > 25:
                break 
            elif category=='bcc' and i > 15:
                break 
            elif category=='bkl' and i > 6:
                break 
            elif category=='df' and i > 85:
                break 
            elif category=='mel' and i > 7:
                break 
            elif category=='vasc' and i >55:
                break 
            elif category=='nv':
                break
            else:
                i += 1
                
# %% Data Preparation

images = []
y = []
CATEGORIES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
label = np.zeros((1,7))
for category in CATEGORIES:
    class_num = CATEGORIES.index(category)
    label[0,class_num] = 1
    image_paths = glob.glob( 'Data_Analysis/'+category+ '/*.jpg' )
    for imagefile in image_paths:
        img  = cv2.imread(imagefile)
        img = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
        h,w,d = img.shape
        img = img/255
        images.append(img)
        y.append(label)
    label = np.zeros((1,7))
    
x = np.array(images)
x1=x.reshape(-1,h,w,d)
y = np.array(y)
y=y.reshape(len(y),7) 
train_img,test_img,train_label,test_label = train_test_split(x,y,test_size=0.25,random_state=42)
# %% Model 1
input_shape=(h,w,d)

model=Sequential()


model.add(Conv2D(64,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(64,(2,2),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


model.add(Conv2D(128,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(128,(2,2),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))

model.add(Conv2D(256,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(256,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(256,(2,2),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(512,(2,2),input_shape=input_shape,activation='relu'))
model.add(Conv2D(512,(2,2),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',Recall()])


model.summary()
# %% Training
early=EarlyStopping(monitor='accuracy',patience=4,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1,cooldown=0, mode='auto',min_delta=0.0001, min_lr=0)
hist = model.fit(train_img,train_label,epochs=500,batch_size=25,validation_data=(test_img, test_label))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['Training', 'validation'])
model.save( '4_128_128_500.h5')

# %% Inference 

test_labels = []
model = load_model('3_128_128_78.h5')
image_paths = glob.glob( 'test/*.jpg' )
for imagefile in image_paths:
    img  = cv2.imread(imagefile)
    img = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
    img = img/255
    np_final = np.expand_dims(img,axis=0)
    pred = model.predict(np_final)
    index = np.argmax(pred[0,:])
    if index == 0:
        labs = "akiec"
    elif index == 1:
        labs = "bcc"
    elif index == 2:
        labs = "bkl"
    elif index == 3:
        labs = "df"
    elif index == 4:
        labs = "mel"
    elif index == 5:
        labs = "nv"
    else:
        labs = "vasc"
    test_labels.append(labs)
