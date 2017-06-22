# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:38:17 2017

@author: wogrady
"""

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
with open('./data_training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path= line[0]
    filename = source_path.split('\\')[-1]
    curr_path = './data_training/IMG/'+ filename
    image = cv2.imread(curr_path)
    images.append(image)   
    measurements.append(float(line[3]))


x_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')