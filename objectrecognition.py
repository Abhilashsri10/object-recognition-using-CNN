# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:34:36 2018

@author: Abhilash Srivastava
"""
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils

np.random.seed(7)
epochs=25
lrate=0.01
decay=lrate/epochs
sgd=SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)

#loading the data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255.0
x_test=x_test/255.0

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

for i in range(0,9):
    plt.subplot(330 +1+i)
    plt.imshow(toimage(x_train[i]))
plt.show()
#setting up the network
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=32)

scores=model.evaluate(x_test,y_test,verbose=0)
print('accuracy: %.2f%%'%(scores[1]*100))