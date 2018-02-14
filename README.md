# RoboND-DeepLearning-Project
Project 4 for Udacity Nanodegree Program

## Introduction
The fourth and final project in the Udacity term 1 Nanodegree program is built around a hypothetical scenario in which an individual is to be tracked by a drone. Some of the constraints, complications to solving this problem are as follows:
 
 - The drone is tracking the person passively (i.e. there is no transmitter or other device to help the drone locate the person)
 - The drone only has a gimbled color 2D camera as a sensor output
 - The person to be tracked (called the 'hero' in the documentation) is not alone, but is mixed in with other people
 - The environment is populated with other non-people objects such as trees, buildings and textured surfaces

To solve this problem, the solution presented will be to build a fully convolutional neural network (FCN) to locate the hero. The primary  advantage that FCN's have over convolutnal networks is that this type of classifier will not only be able to predict the presence of the hero, but also where the hero is in the image.

## Requirements
 1. Writeup / Readme: This document
    - Approach / Architecture
    - Parameters Used
    - Results
    - Future Enhancements
 
 
 



The write-up / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled. The write-up should include a discussion of what worked, what didn't and how the project implementation could be improved going forward.

This report should be written with a technical emphasis (i.e. concrete, supporting information and no 'hand-waiving'). Specifications are met if a reader would be able to replicate what you have done based on what was submitted in the report. This means all network architecture should be explained, parameters should be explicitly stated with factual justifications, and plots / graphs are used where possible to further enhance understanding. A discussion on potential improvements to the project submission should also be included for future enhancements to the network / parameters that could be used to increase accuracy, efficiency, etc. It is not required to make such enhancements, but these enhancements should be explicitly stated in its own section titled "Future Enhancements".
