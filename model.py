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

from keras.callbacks import ModelCheckpoint, CSVLogger


def normalize(data):
    return data / 255 - 0.5

def normalize_img(img):
    img = np.asarray(img)
    img_norm = np.asarray(img).copy() * 0
    cv2.normalize(img, img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm

def augment_lighting(image):
    if random.randint(0, 1) == 1:
        # adjust brightness with random intensity to simulate driving in different lighting conditions 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        image[:,:,2] = image[:,:,2]*random_bright
    return cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

def augment_image_rotation(image, steering):
    if random.randint(0, 1) == 1:
        rows, cols, _ = image.shape
        transRange = 100
        numPixels = 10
        valPixels = 0.4
        transX = transRange * np.random.uniform() - transRange/2
        steering = steering + transX/transRange * 2 * valPixels
        transY = numPixels * np.random.uniform() - numPixels/2
        transMat = np.float32([[1, 0, transX], [0, 1, transY]])
        image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering


def augment_image(image, steering):
    image = augment_lighting(image)
    image, steering = augment_image_rotation(image, steering)
    image, steering = augment_steering(image, steering)
    return image, steering
        

def augment_steering(image, steering):
        index= random.randint(0, 1)
        #index = 1
        if index == 1:
            return cv2.flip(image,1), steering*-1.0
        else:
            return image, steering
    


def generator(samples, batch_size=32):
    #num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        #for offset in range(0, num_samples, batch_size):
        batch_samples = samples[0:batch_size]
        images = []
        measurements = []
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
                assert image is not None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        for image, steering in zip(images, measurements):
            if abs(steering) < 0.0001:
                #count += 1
                #if count % drop_steering_samples == 0:
                if random.randint(0, 1) == 1:
                    aug_image, aug_steering = augment_image(image, steering)
            else:
                aug_image, aug_steering = augment_image(image, steering)
                augmented_images.append(aug_image)
                augmented_measurements.append(aug_steering)
                        
        x_train = np.array(augmented_images)
        y_train = np.array(augmented_measurements)
        #print('Size x_train {0}, Size y_train {1}'.format(len(x_train),len(y_train)))
        yield sklearn.utils.shuffle(x_train, y_train) 

def visualize_steering(lines):
    steering = []
    augmented_steering = []
    count = 0
    for line in lines:
        steering.append(float(line[3]))
        if abs(float(line[3])) < 0.0001:
                count += 1
                if count % drop_steering_samples == 0:
                    augmented_steering.append(float(line[3]))
        else:
            augmented_steering.append(float(line[3]))
    
    plt.figure(figsize=(11, 11))
    plt.xlabel('Steering')
    binwidth = 0.1
    num, bins, patches = plt.hist(steering, bins=np.arange(min(steering) - binwidth, max(steering) + binwidth, binwidth),  normed=1, histtype='bar')
    plt.grid()
    plt.savefig("steering_dist.jpeg", bbox_inches='tight')
    
    plt.figure(figsize=(11, 11))
    plt.xlabel('Steering')
    binwidth = 0.1
    num, bins, patches = plt.hist(augmented_steering, bins=np.arange(min(augmented_steering) - binwidth, max(augmented_steering) + binwidth, binwidth),  normed=1, histtype='bar')
    plt.grid()
    plt.savefig("steering_dist_downsampled_straight.jpeg", bbox_inches='tight')
    

visualize = False
batch_size = 1
drop_steering_samples = 2     
correction = 0.15 # TODO: Tweak to improve  
    

print('Searching for images...')
lines = []
databases = ['./data/driving_log.csv',
            './data_center/driving_log.csv',
             './data_reverse/driving_log.csv',
            './data_swerve/driving_log.csv',
            './data_dirt_bends/driving_log.csv']
#            './data_track_2/driving_log.csv']
#databases = ['./data/driving_log_short.csv']

for location in databases:
    with open(location) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if "steering" not in line:
                lines.append(line)
                
if visualize == True:
    visualize_steering(lines)

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
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

for i in range(0,10):
    test_num = 'test_' + str(i)
    modelName = test_num+'_model_checkpoint.h5'
    modelEnd = test_num+'_model_END.h5'
    csvName = test_num+'_model_checkpoint.csv'
    checkpointer = ModelCheckpoint(filepath=modelName, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csvName)
    
    model.compile(loss='mse', optimizer='adam')
    #model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    #steps_per_epoch=len(np.unique(train_samples))/batch_size
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size, 
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples)/batch_size, 
                                         nb_epoch=3, callbacks=[checkpointer, csv_logger])
    
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
    plt.savefig(test_num+"error_graph.jpeg", bbox_inches='tight')

