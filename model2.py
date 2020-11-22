import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
#df=pd.read_csv(r'/home/gpu/Desktop/fer2013.csv')
#df=pd.read_csv('fer2013.csv')
df=pd.read_csv(r'C:\Users\user\Downloads\mini project\fer2013\fer2013.csv')
df=df[['emotion','pixels','Usage']]

#type(df['emotion'][0])

x_train, y_train, x_test, y_test = [], [], [], []
#a=[]
#for i in range(5):
  #  a.append(df['pixels'][i].split(" "))
#
#print(a[2])

#a= np.array(a, 'float32')
#len(a)     
# df=pd.read_csv(r'/home/gpu/Desktop/fer2013.csv')

for i in range(len(df['Usage'])):
    #df['pixels'][i]=df['pixels'][i].split(" ")
    #df['pixels'][i]=np.array(df['pixels'][i],'float32')
    if 'Training'==df['Usage'][i]:
        y_train.append(df['emotion'][i])
        x_train.append(df['pixels'][i].split(" "))
    if 'PublicTest' in df['Usage'][i]:
        y_test.append(df['emotion'][i])
        x_test.append(df['pixels'][i].split(" "))



x_train= np.array(x_train, 'float32')
#x_train[2]
#x_train=x_train/255
x_test= np.array(x_test, 'float32')
#type(x_test[2][0])
#x_test=x_test/255
import tensorflow as tf

import keras
from keras.utils import to_categorical
len(y_test)
Ny = len(np.unique(y_test))
print(Ny)
y_test = to_categorical(y_test, num_classes = Ny)
y_train = to_categorical( y_train, num_classes = Ny)


x_test.shape


#b = x_train[0] # Shape (28, 28)
#b=b.reshape(48,48)
#b.shape
#plt.imshow(b, cmap='gray')
#plt.show()
#print('Label of image is', y_train[0])


x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')






from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os







num_classes = 7

model = Sequential()
model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48,48,1)))
#model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split = 0.1, epochs=100, batch_size=256)

from keras.models import load_model
#new_model=load_model("WW_untitled0.h5")
model=load_model('model6.h5')
model.summary()




train_score = model.evaluate(x_train,y_train,verbose=0)
print('train loss: ' ,train_score[0])
print('train accuracy: ' ,train_score[1])

test_score = model.evaluate(x_test,y_test,verbose=0)
print('test loss: ' ,test_score[0])
print('test accuracy: ' ,test_score[1])

model.save('model6.h5')



