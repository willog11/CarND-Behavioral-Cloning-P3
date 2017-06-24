# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:38:17 2017

@author: wogrady
"""

import csv
import cv2 
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn


def normalize(data):
    return data / 255 - 0.5

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            correction = 0.25 # TODO: Tweak to improve
            for batch_sample in batch_samples:
                if "steering" not in line:
                    if float(line[6]) < 0.1:
                        continue # Ignore when vehicle is static
                    for i in range(0,3):
                        source_path= line[i]
                        filename = source_path.split('\\')
                        if len(filename) > 2:
                            curr_path = './'+ filename[-3]+'/'+ filename[-2]+'/'+ filename[-1]
                        else:
                            curr_path = './'+'data/'+ filename[-2]+'/'+ filename[-1]
                        print(curr_path)
                        image = cv2.imread(curr_path)
                        images.append(image)
                        if i == 0:
                            measurements.append(float(line[3]))
                        elif i == 1:
                            measurements.append(float(line[3]) + correction) # Left images
                        else:
                            measurements.append(float(line[3]) - correction) # Right images

            # trim image to only see section with road
            images = normalize(np.array(images))
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
                            
            x_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(x_train, y_train) 

print('Searching for images...')
lines = []
databases = ['./data/driving_log.csv',
             './data_center/driving_log.csv',
             './data_reverse/driving_log.csv',
             './data_swerve/driving_log.csv']

for location in databases:
    with open(location) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, 1)
validation_generator = generator(validation_samples, 1)
#train_generator = np.array(train_generator)
#validation_generator = np.array(validation_generator)
#print('Number of training images: {0}, Number of validation images: {1}'.format(str(len(train_generator)), str(len(validation_generator))))

print('Beginning network training...')
model = Sequential()
#model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(((70,0), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


batch_size = 32
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#steps_per_epoch=len(train_samples[0])/batch_size
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')