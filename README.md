# RoboND-DeepLearning-Project
Project 4 for Udacity Nanodegree Program

## Introduction
The fourth and final project in the Udacity term 1 Nanodegree program is built around a hypothetical scenario in which an individual (called the 'hero' in the documentation)  is to be tracked by a drone. Some of the constraints / complications that accompany this problem are as follows:
 
 - The drone is tracking the person passively (i.e. there is no transmitter or other device to help the drone locate the person)
 - The drone only has a gimbled color 2D camera as a sensor for tracking
 - The hero is not alone, but is mixed in with other people
 - The environment is populated with other non-people objects such as trees, buildings and textured surfaces
 - There is no existing model for the hero, the implemented solution must be able to create one from known 'hero positive' images 

To solve this problem, the solution presented will be to build a fully convolutional network (FCN) to locate the hero in the images aquired by the camera. One advantage that FCN's have over convolutnal networks is that this type of classifier will not only be able to predict (or infer) the presence of the hero, but also where the hero is in the image.

## Requirements
 1. Writeup / Readme: This document
    - Approach / Architecture
    - Parameters Used
    - Results
    - Future Enhancements
 2. Model Training Notebook: [model_training.ipynb](model_training.ipynb)
 3. Trained model weights: [model_weights.h5](model_weights.h5)
 4. Model accuracy must be greater than or equal to 40%
 
 
## Approach / Architecture
The approach taken was start with the basic architecture shown in Section 32 for the lab decoder. As covered in that section, a Fully Convolutional Network consists of two major sections: the encoder and the decoder. By reducing the encoder to a 1x1 convolution layer, we can preserve the spatial information that is needed for our application (following the hero). A reproduction of that design (which was used for this submission) is shown below.
![FCN](FCN.png)

Each convolution sweeps its filter(or kernel) across the input one 'step-size' at a time to extract n-features from the image based on the n-filters specified. The output is a convolution layer that is contains stack of all the activated filters. By applying each filter *separately* to each channel of the input, choosing separable convolution layers reduce the number of parameters required. In addition to extracting the features, spatial pooling is also iplemented to reduce the size as it's depth increases and makes the model more scale independent. Increasing the number of filters (depth) will increase the models ability to detect features, but will come at the cost of slowing the model training. 
In the decoder section the feature set is upsampled back to the original image size to output such that we have an output image that pixel for pixel we can map predictions to the outputs. Because the pooling / upsampling done by the FCN is spatially lossy, skip connections are also incorporated into this model  

 
 

The filters were increased iteratively for this submission, and it was found that increasing beyond a depth of 32 and 64 for the filters respectively resulted in the enviroment failing with out of resource errors. Another option for increasing the accuracy of the model was to add additional seperable convolution layers, however since the requirements were met without it, that was not attempted. 

The write-up / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled. The write-up should include a discussion of what worked, what didn't and how the project implementation could be improved going forward.

This report should be written with a technical emphasis (i.e. concrete, supporting information and no 'hand-waiving'). Specifications are met if a reader would be able to replicate what you have done based on what was submitted in the report. This means all network architecture should be explained, parameters should be explicitly stated with factual justifications, and plots / graphs are used where possible to further enhance understanding. A discussion on potential improvements to the project submission should also be included for future enhancements to the network / parameters that could be used to increase accuracy, efficiency, etc. It is not required to make such enhancements, but these enhancements should be explicitly stated in its own section titled "Future Enhancements".
