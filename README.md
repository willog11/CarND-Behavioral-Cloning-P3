# ** Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/NVIDIA_model.png
[image2]:  ./report_images/aug_img_lighting_pre.jpg
[image3]:  ./report_images/aug_img_lighting_post.jpg
[image4]:  ./report_images/aug_img_trans_pre.jpg
[image5]:  ./report_images/aug_img_trans_post.jpg
[image6]:  ./report_images/aug_img_flip_pre.jpg
[image7]:  ./report_images/aug_img_flip_post.jpg
[image8]:  ./report_images/aug_img_all_pre.jpg
[image9]:  ./report_images/aug_img_all_post.jpg
[image10]:  ./report_images/steering_dist.jpeg
[image11]:  ./report_images/steering_dist_downsampled.jpeg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of multiple convolution neural networks with varying filter sizes and depths. The model also includes RELU layers to introduce non-linearity (code line 221-226), and the data is normalized in the model using a Keras lambda layer.  Additionally dropout is used to ensure overfitting does not occur . Further information and code of the model can be found below.

Training was performed by driving the vehicle around the track multiple times and with different driving styles (described further below), also track 2 was used during training to give further data to the network. Not only was various training data collected but augmentation was also applied to further improve on the dataset and reduce over-fitting


#### 2. Attempts to reduce overfitting in the model

Multiple techniques were applied to reduce over-fitting in the model. These techniques ranged from adding a dropout layer to the model and also applying random image augmentation on the training dataset only.

Additionally the mirror cameras are also used too as they offer a different source of information. The correction applied to the steering angle was 0.15, this number was obtained through a trial and error approach because it yielded the best response.

The following  random image augmentation was applied:
1. Changing the brightness of the image to reflect areas of shadows and varying lighting conditions
2. Translating the image in the X and Y direction which simulates hills and creates additional data
3. Flipping the image to balance any biasing due to track layout. Biasing for example can be introduced from track 1 as there are more left turns than right.
4. Down-sampling the straight driving scenarios

In steps 2 and 3 above the steering value (degs) also had to be updated accordingly. The following blocks of code will give a breakdown of how each augmentation technique was applied.
In all cases randint is used very often to create a random probability that augmentation will be to images. 

**Change image lighting**

Each training image had a random lighting augmentation applied. Across all 3 techniques images are first randomly selected for augmentation and also then have a random augmentation applied accordingly. 
This ensures that the network always receives new data and does not over-fitting the data.

~~~
def augment_lighting(image, filename):
    if random.randint(0, 1) == 1:
        if visualize == True and filename != "" and not os.path.isfile(os.getcwd()+"/aug_images/"+filename):
            cv2.imwrite("aug_images/"+filename,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # adjust brightness with random intensity to simulate driving in different lighting conditions 
        aug_image = np.float32(np.copy(image))
        aug_image = cv2.cvtColor(aug_image,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        aug_image[:,:,2] = aug_image[:,:,2]*random_bright
        if visualize == True and filename != "":
            cv2.imwrite("aug_images/aug_img_lighting_"+filename,cv2.cvtColor(aug_image, cv2.COLOR_HSV2BGR))
        return cv2.cvtColor(aug_image,cv2.COLOR_HSV2RGB)
    else:    
        return image
~~~

To augment the lighting conditions the image is converted to HSV and then randomly tweak the V color space. By doing this the image either gets brighter or darker. From there it is converted back to RGB

The image on the left is the original image, whilst the image on the right has been made darker by this augmentation. This is very important as the bonus track contains a lot of shadows whilst the first track is generally very bright.

![alt text][image2] ![alt text][image3]

**Translate the image**

The next step was to shift the image in both X and Y directions.By doing this it simulates hilly conditions which is very prominent in the bonus track. 

Note also at this point the steering needs to be updated as shift per pixel in the x axis. A shift per pixel value of 0.004 was decided upon through trial and error testing.

~~~
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
~~~

As can be seen above a range of 100 pixels in the x axis and 10 in the y was set to limit the amount the image can be shifted. Also np.random.uniform() is used to generate a random value between 0-1 to scale the shifting by.
Finally the image augmentation is applied to both X and Y axis, also the steering is shifted according to our shift per pixel (along x axis) of 0.004.

The following images show this augmentation: Left is pre shift and right is post.

![alt text][image4] ![alt text][image5]

**Flip the image**

The final step of the augmentation was to flip the image and steering angles accordingly. As per before this was all done randomly to further reduce the chance of over-fitting.

~~~
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
~~~

In the following 2 images can be found the pre and post process of this augmentation:
![alt text][image6] ![alt text][image7]

The above 3 steps are combined which results in a model that can use all 3 augmentation techniques. Thus the image can be brightened\darkened, shifted and flipped. The following is an example of this where the image was brightened, shifted right and down and finally flipped. As before, the left image is the original and the right is the final output

~~~
def augment_image(image, steering, filename=""):
    aug_light = augment_lighting(image, filename)
    aug_trans, aug_steer_trans = augment_image_translate(aug_light, steering, filename)
    aug_flip, aug_steer_flip = augment_flip(aug_trans, aug_steer_trans, filename)
    return aug_flip, aug_steer_flip
~~~

![alt text][image8] ![alt text][image9]

**Down-sample straight driving**

Its very clear from driving track 1 that most of the track is straight driving. Thus the resulting model may be biased by this, thus down-sampling as required. As before, this was done using a randomly. The following images show how this down-sampling results in a more uniformly distributed model

~~~
for image, steering in zip(images, measurements):
	if abs(steering) < 0.0001: # Check for straight driving and downsample
		if random.randint(0, 1) == 1:
			aug_image, aug_steering = augment_image(image, steering)
	else:
		aug_image, aug_steering = augment_image(image, steering)
~~~

It results in the following graphs:

![alt text][image10] ![alt text][image11]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from both sides, driving the first track in reverse and recaptures of difficult cases such as corners with dirt edges. Finally I captured 3 laps on the second track as I wanted to ensure it performed well there too. This track is much more difficult as its a very different scenario and lighting conditions. Also the turns are more severe and the roads have much different edges than track 1.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA model I thought this model might be appropriate because it was highly recommended by the lecture sections and also by fellow engineers. To load in the images into the network I used a generator based approach to maximize the memory usage of my GPU (TitanX)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. It was essential that only the training data be augmented so I disabled this for the training data. Also the training and validation was split 80% training and 20% validation where on each case the data was shuffled to ensure new data was always used

~~~
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size, True)
validation_generator = generator(validation_samples, batch_size, False)

def generator(samples, batch_size=32, augment_data=False):
	while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
		......
~~~

To combat the overfitting, I modified the model so that it also includes dropout layers. That coupled with the augmented data ensures that over-fitting is not an issue.

The final step was to run the simulator to see how well the car was driving around track one. At first the vehicle did not perform well on the track, this was due to a few underlying bugs which I missed in my implementation. Upon correcting these the vehicle performed very well on both tracks

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model is essentially the NVIDIA architecture with the addition of a dropout layer to help prevent over-fitting. However before any of that can happen the algorithm first iterates through all log csv files and extracts the images and corresponding steering data.
The images by from the simulator are in BGR format so must be converted to RGB. Additionally to use the additional mirror cameras the steering angle must be updated accordingly. This will be discussed further below.
The actual NVIDIA architecture is as follows

![alt text][image1]

The actual implementation was as follows:

| Layer | Logic         		| Description	        					| 
|:-----:|:---------------------|:--------------------------------------------------| 
| **1**	  | Input         		| Mean centered nomalization - 160x320x3 BGR image   | 
| **2**  | Cropping         		| Remove top part of image (sky removal)				| 
| **2**   | Convolution 5x5     	| 2x2 stride, VALID padding, outputs 24 layers	|
|	     | RELU					| RELU activation based on the convulational output	|
| **3**	     | Convolution 5x5	    | 2x2 stride, VALID padding, outputs 36 layers    									|
|	     | RELU					| RELU activation based on the convulational output	|
| **4**	     | Convolution 5x5	    | 2x2 stride, VALID padding, outputs 48 layers    									|
|	     | RELU					| RELU activation										|
|	     | Dropout					|	Drop 50% randomly of the RELU output						|
| **5**	  | Convolution 5x5	    | 3x3 stride, VALID padding, outputs 64 layers    									|
|	     | RELU					| RELU activation										|
| **6**	  | Convolution 5x5	    | 3x3 stride, VALID padding, outputs 64 layers    									|
| **7**	     | Flatten			| Flaten the feature map, Output =1164. 								|
| **8**	     | Fully Connected				| Input = 1164 Output = 100 									|
| **9**	     | Fully Connected				| Input = 100 Output = 50 									|
| **10**	     | Fully Connected				| Input = 50 Output = 10 									|
| **11**	     | Fully Connected				| Input = 10 Output = 1 									|

The code of which is as follows:

~~~
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
~~~


#### 3. Creation of the Training Set & Training Process

The previous sections describe well the capturing of the training set and how it was shuffled and split 80/20 training and validation set. The size of which was training set size of 15,109 (without using mirror cameras, thus 45,327 images overall) and validation set size 3778

Finally when training the model I created a simple loop which would train the model 10 times, each time saving the best model, final model and plot of the reduction in error. This was very useful because due to the nature of random probability the same built model can give different results when built multiple times. Thus by building 10 times I can check each model and verify the best one. Note the second built model resulted in the vehicle driving around both tracks very well. Thus it was selected as the final model.

The code for this is very straight forward and is as follows:

~~~
# Tunable parameters for algorithm
visualize = False
batch_size = 32  
steer_correction = 0.15 

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
~~~

As it can be seen only 5 epochs were required to create a functioning network. However bare in mind that 45,327 images were actually used in the model so each epoch even on a TitanX GPU took aprox 3mins (15-20mins to build a model). Also the adam optimiser was used to generate the mean squared error (mse). This was a very effective approach as can be seen in the resulting model.

In the current folder can be found track1.mp4 and track2.mp4 which are the recordings from the replayed network. These videos show the very good performance of the model from the front camera.

For ease of demo I have also uploaded the track videos on YouTube and added the thumbnails below

**Track 1 Demo**

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/https://youtu.be/32Ep9985M0Y/0.jpg)](http://www.youtube.com/watch?v=https://youtu.be/32Ep9985M0Y)


**Track 2 Demo**

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/https://youtu.be/IUREPTEzMMU/0.jpg)](http://www.youtube.com/watch?v=https://youtu.be/IUREPTEzMMU)


