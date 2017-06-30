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
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Dropout, Lambda
from sklearn.model_selection import train_test_split
import sklearn
import os.path

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

def augment_lighting(image, filename):
    if random.randint(0, 1) == 1:
        if visualize == True and filename != "" and not os.path.isfile(os.getcwd()+"/aug_images/"+filename):
            cv2.imwrite("aug_images/"+filename,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        aug_image = np.float32(np.copy(image))
        aug_image = cv2.cvtColor(aug_image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        aug_image[:,:,2] = aug_image[:,:,2]*random_bright
        if visualize == True and filename != "":
            cv2.imwrite("aug_images/aug_img_lighting_"+filename,cv2.cvtColor(aug_image, cv2.COLOR_HSV2BGR))
        return cv2.cvtColor(aug_image,cv2.COLOR_HSV2RGB)
    else:    
        return image

def augment_image_translate(image, steering, filename):
    if random.randint(0, 1) == 1:
        if visualize == True and filename != "" and not os.path.isfile(os.getcwd()+"/aug_images/"+filename):
            cv2.imwrite("aug_images/"+filename,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        aug_image = np.float32(np.copy(image))
        rows, cols, _ = aug_image.shape
        trans_range_x = 100
        trans_range_y = 10
        steer_shift_px = 0.004
        trans_x = trans_range_x * np.random.uniform() - trans_range_x/2
        aug_steering = steering + trans_x * steer_shift_px
        trans_y = trans_range_y * np.random.uniform() - trans_range_y/2
        trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        aug_image = cv2.warpAffine(aug_image, trans_mat, (cols, rows))
        if visualize == True and filename != "":
            cv2.imwrite("aug_images/aug_img_trans_"+filename,cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        return aug_image, aug_steering
    else:
        return image, steering


def augment_image(image, steering, filename=""):
    aug_light = augment_lighting(image, filename)
    aug_trans, aug_steer_trans = augment_image_translate(aug_light, steering, filename)
    aug_flip, aug_steer_flip = augment_flip(aug_trans, aug_steer_trans, filename)
    return aug_flip, aug_steer_flip
        

def augment_flip(image, steering, filename):
    if random.randint(0, 1) == 1:   
        if visualize == True and filename != "" and not os.path.isfile(os.getcwd()+"/aug_images/"+filename):
            cv2.imwrite("aug_images/"+filename,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))    
        aug_image = np.float32(np.copy(image))    
        aug_image = cv2.flip(aug_image,1)
        aug_steering = steering * -1
        if visualize == True and filename != "":
            cv2.imwrite("aug_images/aug_img_flip_"+filename,cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        return aug_image, aug_steering
    else:
        return image, steering
        
def get_img(line):
    source_path= line
    filename = source_path.split('\\')
    if len(filename) > 2:
        curr_path = './'+ filename[-3]+'/'+ filename[-2]+'/'+ filename[-1]
    else:
        curr_path = './'+'data/'+ filename[-2]+'/'+ filename[-1]
    image = cv2.imread(curr_path)
    assert image is not None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) , filename[-1]
    
    


def generator(samples, batch_size=32, augment_data=False):
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        batch_samples = samples[0:batch_size]
        images = []
        measurements = []
        for batch_sample in batch_samples:
            if float(batch_sample[6]) < 0.1:
                continue # Ignore when vehicle is static
            for i in range(0,3):
                image, filename = get_img(batch_sample[i])
                images.append(image)
                if i == 0:
                    measurements.append(float(batch_sample[3]))
                elif i == 1:
                    measurements.append(float(batch_sample[3]) + steer_correction) # Left images
                else:
                    measurements.append(float(batch_sample[3]) - steer_correction) # Right images

        images = np.array(images)
        if augment_data == True:
            augmented_images, augmented_measurements = [], []
            for image, steering in zip(images, measurements):
                if abs(steering) < 0.0001: # Check for straight driving and downsample
                    if random.randint(0, 1) == 1:
                        aug_image, aug_steering = augment_image(image, steering)
                else:
                    aug_image, aug_steering = augment_image(image, steering)
                    augmented_images.append(aug_image)
                    augmented_measurements.append(aug_steering)
                            
            x_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
        else:
            x_train = np.array(images)
            y_train = np.array(measurements)
        #print('Size x_train {0}, Size y_train {1}'.format(len(x_train),len(y_train)))
        yield sklearn.utils.shuffle(x_train, y_train) 

def visualize_steering(lines):
    steering = []
    augmented_steering = []
    for line in lines:
        steering.append(float(line[3]))
        if abs(float(line[3])) < 0.0001:
                if random.randint(0, 1) == 1:  
                    augmented_steering.append(float(line[3]))
        else:
            augmented_steering.append(float(line[3]))
    
    plt.figure(figsize=(11, 11))
    plt.xlabel('Steering')
    plt.title('Original Steering Distribution')
    binwidth = 0.1
    num, bins, patches = plt.hist(steering, bins=np.arange(min(steering) - binwidth, max(steering) + binwidth, binwidth),  normed=1, histtype='bar')
    plt.grid()
    plt.savefig("steering_dist.jpeg", bbox_inches='tight')
    
    plt.figure(figsize=(11, 11))
    plt.xlabel('Steering')
    plt.title('Down-sampled Steering Distribution')
    binwidth = 0.1
    num, bins, patches = plt.hist(augmented_steering, bins=np.arange(min(augmented_steering) - binwidth, max(augmented_steering) + binwidth, binwidth),  normed=1, histtype='bar')
    plt.grid()
    plt.savefig("steering_dist_downsampled.jpeg", bbox_inches='tight')

def visualize_aug_images(lines, batch_size = 5): 
    directory = "aug_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = sklearn.utils.shuffle(lines)
    batch_samples = samples[0:batch_size]
    for batch_sample in batch_samples:
        for i in range(0,3):
            image, filename = get_img(batch_sample[i])
            augment_image(image, 0, filename)
        
# Tunable parameters for algorithm
visualize = True
batch_size = 32  
steer_correction = 0.15 
    

print('Searching for images...')
lines = []
databases = ['./data/driving_log.csv',
            './data_center/driving_log.csv',
             './data_reverse/driving_log.csv',
            './data_swerve/driving_log.csv',
            './data_dirt_bends/driving_log.csv',
            './data_track_2/driving_log.csv',
            './data_track_2_t2/driving_log.csv']

for location in databases:
    with open(location) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if "steering" not in line:
                lines.append(line)
                
if visualize == True:
    visualize_steering(lines)
    visualize_aug_images(lines)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print('Trainging set size {0}, Validation set size {1}'.format(len(train_samples), len(validation_samples)))
train_generator = generator(train_samples, batch_size, True)
validation_generator = generator(validation_samples, batch_size, False)

print('Beginning network training...')
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(((70,0), (0,0))))
model.add(Convolution2D(24, 5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Convolution2D(64, 3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

for i in range(0,10):
    print('####################################')
    print('Training Test Case {0}'.format(i))
    print('####################################')
    test_num = 'test_' + str(i)
    modelName = test_num+'_model_checkpoint.h5'
    modelEnd = test_num+'_model_END.h5'
    csvName = test_num+'_model_checkpoint.csv'
    checkpointer = ModelCheckpoint(filepath=modelName, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csvName)
    
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size, 
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples)/batch_size, 
                                         nb_epoch=5, callbacks=[checkpointer, csv_logger])
    
    model.save(modelEnd)
    model.save(modelName)
    
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

