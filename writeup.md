# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model-mean-squared-error-loss]: ./images/model-mean-squared-error-loss.png "Mean square error loss"
[center-driving]: ./images/center-driving.jpg "Center driving"
[recovery-1]: ./images/recovery-1.jpg "Recovery driving 1"
[recovery-2]: ./images/recovery-2.jpg "Recovery driving 2"
[recovery-3]: ./images/recovery-3.jpg "Recovery driving 3"
[class-distribution]: ./images/class-distribution.png "Steering class distribution"
[model-architecture]: ./images/model-architecture.png "Model architecture"
[left-image]: ./images/left-image.jpg "Left driving image"
[right-image]: ./images/right-image.jpg "Right driving image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 Video recording of autonomous driving around track 1
* video2.mp4 Video recording of autonomous driving around track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training, saving and evaluating the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The chosen model is the model described in the NVIDIA deep learning self driving cars blog @ https://devblogs.nvidia.com/deep-learning-self-driving-cars/. The NVIDIA model consists of 5 convolution layers and 3 connected layers.

Additionally, to normalize the data, the image value are divided by 255 (with 255 being the maximum value for the color space).

#### 2. Attempts to reduce overfitting in the model

In my experiments, running the model with a low number epochs yield little overfitting. As such, the model was run with 3 epochs. The model loss can be visualized below, illustrating that the model was not overfitting.

![model-mean-squared-error-loss]

After training the model with 3 epochs, the model was tested through the simulator to check that the vehicle could stay within the track.

Additionally tests were ran to reduce overfitting by introducing batch normalization and dropouts with a high epoch. However, this resulted in a worst off model.

#### 3. Model parameter tuning

As an adam optimizer was used the learning rate was not tuned manually.

#### 4. Appropriate training data

My training data from the simulator included:
* Center driving around the track
* Augmented data by flipping the image and steering measurement:
* Augmented data by applying random brightness to the images

For details about how I created the training data, see the next section. 

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with the image recognition models recommended in the course, starting with the LeNet model followed by the NIVIDA model.

The first step was to implement and experiment with the LeNet model.

To evaluate the model, my driving log data was split into a training, validation and testing set.

The LeNet model yield a low mean squared error on the training and validation set.

The next step was the evaluate both models to see how well the car drives around the first track. In my test with the LeNet model, the car fell off the track.

My next step was to increase the number of epochs and modify the batch size. However, the increase in epochs resulted overfitting due to the relatively larger mean squared error on the validation set. Modifications to the batch size did not seem to improve the test in the simulator.

As such the NIVIDIA model was implemented next. Compared to the LeNet model, the NIVIDIA model included 3 additional convolution layers and 1 connected layer.

Once the model have been implemented, I experimented with variations in the epochs and batch size. I found that my optimal epochs and batch size was 3 and 64 respectivelyw which resulted in both the model not overfitting and the vehicle being able to drive around the track without leaving the road.

#### 2. Final Model Architecture

As depicted in the table below, the final model architecture in based on the NIVIDA model.

| Layer (type)               | Output Shape            | Param # |
|----------------------------|-------------------------|---------|
| lambda_1 (Lambda)          | (None, 160, 320, 3)     | 0       |
| cropping2d_1 (Cropping2D)  | (None, 90, 320, 3)      | 0       |
| conv2d_1 (Conv2D)          | (None, 43, 158, 24)     | 1824    |
| activation_1 (Activation)  | (None, 43, 158, 24)     | 0       |
| conv2d_2 (Conv2D)          | (None, 20, 77, 36)      | 21636   |
| activation_2 (Activation)  | (None, 20, 77, 36)      | 0       |
| conv2d_3 (Conv2D)          | (None, 8, 37, 48)       | 43248   |
| activation_3 (Activation)  | (None, 8, 37, 48)       | 0       |
| conv2d_4 (Conv2D)          | (None, 6, 35, 64)       | 27712   |
| conv2d_5 (Conv2D)          | (None, 4, 33, 64)       | 36928   |
| flatten_1 (Flatten)        | (None, 8448)            | 0       |
| dense_1 (Dense)            | (None, 100)             | 844900  |
| dense_2 (Dense)            | (None, 50)              | 5050    |
| dense_3 (Dense)            | (None, 10)              | 510     |
| dense_4 (Dense)            | (None, 1)               | 11      |

The first layer, `lambda_1` is a lambda layer that 
* Normalizes the input value between 0 and 1 by dividing the input value by 255.
* Centres the input values by subtracting 0.5 from the normalized value.

As we are focusing on only the road, the second layer, `cropping2d_1` crops out the top and bottom of the image.

This is followed by the implementation of the NVIDIA model which includes 5 convolution layers and 3 connected layers.

Finally a single output connected layer, `dense_4` is applied as we are only predicting a single steering output value.

Here is a visualization of the architecture

![alt text][model-architecture]

#### 3. Creation of the Training Set & Training Process

In my experiments with gathering of the training data, I noticed that using a mouse or keyboard ended up with lots of zero steering value even on a curve. To improve my training data, I used a steering wheel with the simulator. This yield much more accurate steering data around the curve roads.

My training data included:
* 3 center lane driving laps around track 1
* 1 lap in reverse around track 1

To capture the center land driving, I recorded three laps on track one driving up the center of the road. Here is an example image of center lane driving:

![alt text][center-driving]

As suggested by the course, I experimented with implemented recovery from the left and right side of the road to the center. However including these image in the data did not change the behavior of the driving in the simulator. Below are images illustrating what a recovery looks like starting from the left:

![alt text][recovery-1]
![alt text][recovery-2]
![alt text][recovery-3]

Then I repeated this process on track two in order to get more data points.

Next, an analysis of the data showed that the track has a bias steering to one side. As such, added augmented data by duplicating the existing data and flipping the images and steering angles and adding it to the training set. This yield a more normalized steering data set as depicted below.

![alt text][class-distribution]

Furthermore, to ensure that the data is more generalized, I added the left and right images obtain from the driving logs and added a steering correction for the left and right image labels.


Example of right steering image:

![Right driving image][right-image]


Example of left steering images:

![Left driving image][left-image]


From the image above, we can see that in the right steering image, the car is closer to the right line, thus it requires a steering correction to steer the car more to the left. The inverse logic applies to the left steering image. The calculation for the left and right steering is as follows:

> right steering = center steering - steering correction

> left steering = center steering + steering correction

I also implemented a random brightness augmentation to the images to make the model more robust. 

Including the flipped data, the data set totaled tally was 58230. 

The data was then shuffled and split into the following ratio using the `train_test_split()` function:
* 80% training
* 15% validation
* 5% test

To reduce the memory usage by not storing training data in memory, a generator was implemented. The generator 
* Returns the a random samples for a specified batch size
* Applies the data augmentation (flip and random brightness) to the data 

Finally, the generator for both the training and validation is passed into the keras `fit_generator` function to train the model.

In conclusion, my chosen parameters are:
* Epoch: 3
* Batch Size: 64
* Steering correction (for left and right image): 0.2

With the above parameters, I manage to train a model that will drive the car within the road for track 1. A recording of the car driving within the road can be found in the [video.mp4](video.mp4) file.

### Track 2
 In order to get the model working for track 2, I gather additional data.

 1. 2 laps around track 2 driving in the right lane
 2. 1 lap around track 2 in reverse driving in the right lane
 3. Recovery driving for sections whereby the car drove off the road

 I also updated the drive.py speed to 10 as I had an issue with the car getting stuck when going down hill. Seems like there may be a bug with the car stopping permanently if it is going too slow.

 A recording of the car driving within the road can be found in [video2.mp4](video2.mp4). 
