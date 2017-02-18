#**Traffic Sign Recognition** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/preprocessing.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Readme.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

####4. Video
https://youtu.be/AQgWwrdXUIM

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 128
(model.py lines 108 the function create_model) 

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a
 Keras lambda layer. Furehtermore, the input image data is cropped to keep the image regions that
 are relevant to the driving task (containing lane lines between row ).

####2. Attempts to reduce overfitting in the model

I started using dropout layers but noticed that they can drop some scarce data leading the car to not perform well.
The situation I had using Udacity dataset was where the car drives well in 90% of the track
but fails to detect and steer when confronted to a turn with no lane lines (brown mud in the second turn). Therefore
I decided to remove the dropout layer. An alternative (probably better) approach would be to collect more data in such 
crucial points in order to regulate the training data.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (function create_model).

####4. Appropriate training data

I tried collecting my own data, did dozens of laps and different driving maneuvers. I couldn't seem to get
 data that was good enough. Driving with the keyboard was bad anyway, the mouse made it a little bit better. I think
 the problem was that this task is kind of ill conditioned. We try to train a car to drive in a certain way by associating
 camera images and steering angles. The bad conditioning comes from the fact there is many different ways of driving
 the circuit and it is almost impossible for a human being to drive a circuit twice in an exact manner. This would lead
 the CNN to try training an incoherent model, resulting in a validation error that gets stuck at some point and can't
 be reduced anymore (in my case around 0.01).
 
 Also that validation score doesn't correlate with the quality of the autonomous driving in any way.
 
 I decided to focus on the data provided by Udacity and preprocess it.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to check what people are already doing in the litterature
and try to buld up something similar. That would be the basis from where
I start to fine-tune the model.

After I collected data myself by driving few laps. I ran my training

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle would come close to the borders or even touch. Playing around with the angle
offset parameter (used to augment data from left and right cameras associationg them with a modified steering angle)
helped a bit. Also running the training again with more epochs reduced the wobbling of the car.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers
 
Cropping Layer: 50 pixels from top (sky) and 20 pixel from bottom (car) are cropped

Normalization Layer

Convolution2D: filterdepth 6, 5x5 convolution
Convolution2D: filterdepth 36, 5x5 convolution, subsampling by 2x2
Convolution2D: filterdepth 48, 5x5 convolution, subsampling by 2x2
Convolution2D: filterdepth 64, 3x3 convolution

Flattening Layer

Dense Layer 1024
Dense Layer 128
Dense Layer 64
Dense Layer 16
Dense Layer 1 (output)

All layers had relu activations
And finally the Adam optimizer with mean square error loss function.


####3. Creation of the Training Set & Training Process

I recorded all kinds of data as suggested:

driving centrally for few laps

driving in and out from the center and recording only when driving towards the center

Focusing on special spots and doing avoiding maneuvers

Using all those combinations, I couln't really get the car to drive well

Then I decided to use Udacity's datasets. The resulting car behaviour was better.

Some preprocessing was necessary though.

I drew a histogram of the steering angles provided by the Udacity dataset and noticed that there are overwhelmingly more
data samples with a steering angle equal to zero. Therefore, I decided to discard 90% of those points.

I used the image provided by left and right cameras as fake center images where I added a steering offset.
For a given center image with a steering angle x, I augmented the data with the left image associated with a 
steering angle x + offset and the right image associated with a steering angle x - offset. The offset I chose in my
case was 0.1.

Here are the histograms that show the resulting distribution of training samples after preprocessing:

![alt text][image1]


To further augment the data set, I flipped images and angles thinking that this would simulate the car recovering
path from both left and right sides.

This step is done on the generator level (function data_generator) And therefore I had to be careful that my dataset size
would be doubled (which expalains the multiplication by 2 in line 169 and 170)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
After the collection process, I had around 20000 number of data points.

I used this training data for training the model. The validation set wasn't really helpful as it didn't really describe
how well the car would do on the track. But I could at least make sure that my model wasn't overfitting.
The ideal number of epochs was not more than 5 as the validation score didn't decrease after that.
I used an adam optimizer so that manually training the learning rate wasn't necessary.