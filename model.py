# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:38:17 2017

@author: wogrady
"""

import csv
import cv2 
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D


def normalize(data):
    return data / 255 - 0.5

print('Searching for images...')
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    if "steering" not in line:
        source_path= line[0]
        filename = source_path.split('\\')[-1]
        curr_path = './data/'+ filename
        image = cv2.imread(curr_path)
        images.append(image)   
        measurements.append(float(line[3]))
        
print('Number of images added: {0}'.format(str(len(images))))
print('Augmenting images...')
images = normalize(np.array(images))

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
print('Number of total images including augmented: {0}'.format(str(len(augmented_images)))) 
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('Beginning network training...')
model = Sequential()
#model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')