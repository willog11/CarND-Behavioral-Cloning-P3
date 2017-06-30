# ** Behavioral Cloning** 


---

** Behavioral Cloning Project**

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
[image4]:  ./report_images/aug_img_trans_pre.png
[image5]:  ./report_images/aug_img_trans_post.png
[image6]:  ./report_images/NVIDIA_model.png
[image7]:  ./report_images/NVIDIA_model.png

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

My model is essentially the NVIDIA architecture with the addition of a dropout layer to help prevent over-fitting. However before any of that can happen the algorithm first iterates through all log csv files and extracts the images and corresponding steering data.
The images by from the simulator are in BGR format so must be converted to RGB. Additionally to use the additional mirror cameras the steering angle must be updated accordingly. This will be discussed further below.
The actual NVIDIA architecture is as follows

![alt text] [image1]

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


#### 2. Attempts to reduce overfitting in the model

Multiple techniques were applied to reduce over-fitting in the model. These techniques ranged from adding a dropout layer to the model and also applying random image augmentation on the training dataset only.

Additionally the mirror cameras are also used too as they offer a different source of information. The correction applied to the steering angle was 0.15, this number was obtained through a trial and error approach because it yielded the best response.

The following  random image augmentation was applied:
1. Changing the brightness of the image to reflect areas of shadows and varying lighting conditions
2. Translating the image in the X and Y direction which simulates hills and creates additional data
3. Flipping the image to balance any biasing due to track layout. Biasing for example can be introduced from track 1 as there are more left turns than right.

In steps 2 and 3 above the steering value (degs) also had to be updated accordingly. The following blocks of code will give a breakdown of how each augmentation technique was applied.
In all cases randint is used very often to create a random probability that augmentation will be to images. 

** Change image lighting **
Each training image had a random lighting augmentation applied. Across all 3 techniques images are first randomly selected for augmentation and also then have a random augmentation applied accordingly. 
This ensures that the network always receives new data and does not overfitting the data.

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

The image on the left is the original image, whislt the image on the right has been made darker by this augmentation. This is very important as the bonus track contains a lot of shadows whilst the first track is generally very bright.
![alt text][image2] ![alt text][image3]

** Translate the image **

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

The following images show this augmentation: Left is pre shift and right is post
![alt text][image4] ![alt text][image5]

** Flip the image **

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
