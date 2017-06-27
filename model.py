# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:38:17 2017

@author: wogrady
"""

import csv
import cv2 
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Dropout, Lambda, BatchNormalization
from sklearn.model_selection import train_test_split
import sklearn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint


def normalize(data):
    return data / 255 - 0.5

def normalize_img(img):
    img = np.asarray(img)
    img_norm = np.asarray(img).copy() * 0
    cv2.normalize(img, img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm

def generator(samples, batch_size=32):
    #num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        #for offset in range(0, num_samples, batch_size):
        batch_samples = samples[0:batch_size]
        images = []
        measurements = []
        correction = 0.25 # TODO: Tweak to improve
        for batch_sample in batch_samples:
            if float(batch_sample[6]) < 0.1:
                continue # Ignore when vehicle is static
            for i in range(0,3):
                source_path= batch_sample[i]
                filename = source_path.split('\\')
                if len(filename) > 2:
                    curr_path = './'+ filename[-3]+'/'+ filename[-2]+'/'+ filename[-1]
                else:
                    curr_path = './'+'data/'+ filename[-2]+'/'+ filename[-1]
                image = cv2.imread(curr_path)
                images.append(image)
                if i == 0:
                    measurements.append(float(batch_sample[3]))
                elif i == 1:
                    measurements.append(float(batch_sample[3]) + correction) # Left images
                else:
                    measurements.append(float(batch_sample[3]) - correction) # Right images

        # trim image to only see section with road
        #images = normalize(np.array(images))
        images = np.array(images)
        augmented_images, augmented_measurements = [], []
        count = 0
        for image, measurement in zip(images, measurements):          
            if abs(measurement) < 0.0001:
                count += 1
                if count % 1 == 0:
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
            else:
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                index= random.getrandbits(1)
                index = 1
                if index == 1:
                    augmented_images.append(cv2.flip(image,1))
                    augmented_measurements.append(measurement*-1.0)
                        
        x_train = np.array(augmented_images)
        y_train = np.array(augmented_measurements)
        #print('Size x_train {0}, Size y_train {1}'.format(len(x_train),len(y_train)))
        yield sklearn.utils.shuffle(x_train, y_train) 

print('Searching for images...')
lines = []
databases = ['./data/driving_log.csv',
            './data_center/driving_log.csv',
             './data_reverse/driving_log.csv',
            './data_swerve/driving_log.csv',
            './data_dirt_bends/driving_log.csv']
#databases = ['./data/driving_log_short.csv']

for location in databases:
    with open(location) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if "steering" not in line:
                lines.append(line)

batch_size = 1
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)
#train_generator = np.array(train_generator)
#validation_generator = np.array(validation_generator)
#print('Number of training images: {0}, Number of validation images: {1}'.format(str(len(train_generator)), str(len(validation_generator))))

print('Beginning network training...')
model = Sequential()
#model.add(BatchNormalization(axis=1, input_shape=(160,320,3)))
#model.add(Lambda(normalize, input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(((70,0), (0,0))))
model.add(Convolution2D(24, 5,5,subsample=(2,2), activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(36, 5,5,subsample=(2,2), activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(48, 5,5,subsample=(2,2), activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, activation='relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, activation='relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


modelName ='model_checkpoint.h5'
modelEnd = 'model_END.h5'
checkpointer = ModelCheckpoint(filepath=modelName, verbose=1, save_best_only=True)

model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#steps_per_epoch=len(np.unique(train_samples))/batch_size
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size, 
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples)/batch_size, 
                                     nb_epoch=3, callbacks=[checkpointer])

model.save(modelEnd)
model.save(modelName)
### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure(figsize=(11, 11))
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig("error_graph.jpeg", bbox_inches='tight')

