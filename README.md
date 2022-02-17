# Anomalous Dice

## Table of Contents
1. [Description](#description)
2. [Data](#data)
  1. [Preprocessing](#preprocessing)
	2. [Rotations](#rotations)
  3. [Thresholds](#thresholds)
3. [Anomaly Detection](#anomaly-detection)
  1. [Numpy Approach](#numpy-approach)
	2. [Auto Encoder](#AutoEncoder)
  3. [Auto Encoder with Classifier](#AutoEncoder-with-classifier)
  4. [Variational Auto Encoder](#VAE)
4. [Installation](#installation)
5. [Tech](#tech)

## Description

The goal of this project is detecting visual anomalies without knowing what the anomalous look
like. 

## Data

### Preprocessing

We started this project by looking at the data provided. The image dataset contains 6571 images
of dice facing towards the camera. They are 128x128 and divided into 11 classes. Each of the
6 faces has several directions in which they can be oriented. This explains why there are
so many classes.

![](visuals/dataset.jpg)

A first step was to edit these images so they can be divided into only 6 classes. One for each face.
That would in the long run save us inference time of any running model, because less comparisons 
or generations would need to be made for training and predicting.

### Rotations

In order to get these dice facing the correct direction a method was used that draws a line halfway 
along the image and counts the amount of dark pixels. We then rotate the image by 1 degree and repeat.
When doing this for all the diagonals we, in the end, can select the one with the most dark pixels,
and rotate in this way. 

![](visuals/dice-perprocessing.gif)

For doing this we had to crop circles out of our images in order to avoid misclassifications with the 
edge of the image.

From these templates were made by squashing the numpy arrays together.

1                           |2                           |3                           |4                           |5                           |6                           |
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)

You can clearly see this approach was not perfect in the last template. This could be solved by adjusting
thresholds but in the interest of saving time it was left to a later stage.

This approach did have an added benefit that anomalies would actually affect the symmetry of the result.
In the following examples you can see that the anomalies are sometimes rotated in a way that differs from 
how normal dice are rotated. When comparing these values to the templates, we can immediately recognize a
number of anomalies without any modeling.

![](visuals/ano3.gif)
![](visuals/ano4.gif)

### Thresholds

After this stage images of all classes were compared to the templates and using loss functions the differences
would be calculated for each of these templates. Doing this gave an idea of whithin what range a correct classification 
would be. and these were then used as thresholds in later stages of the project.

This is an example using MSEloss

category 1    |category 2    |category 3    |category 4    |category 5    |category 6    |
:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
62.02         |60.43         |66.30         |69.93         |74.30         |82.75         |


## Anomaly Detection

We tried different approaches for Anomaly Detection.

### Numpy approach

For this approach we relied heavily on the thresholds calculated in the previous chapter.
When a prediction is made on a sample, that sample is compared to each of the thresholds. 
If it falls within the boundaries of one of these it is calculated as a normal sample.

But when a sample like this one is given: 

![](processed_data/test_set/ano/17_11_21_anomalies_005.png)

![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
65.51   > 62.02             |75.41   > 60.43             |67.54   >66.30              |72.45   > 69.93             |77.51   > 74.30             |91.54  > 82.75     
MSEloss > thresh 1          |MSEloss > thresh 2          |MSEloss > thresh 3          |MSEloss > thresh 4          |MSEloss > thresh 5          |MSEloss > thresh 6  

And MSEloss for this sample falls outside of the boundaries for each category it is
classified as an anomaly.

metric|score
:--------------------------:|:--------------------------:
f1|0.90 
Accuracy|0.90 

### AutoEncoder 

We chose to work with an autoencoder because we believed it would be suited to the
task. Rather than using a Generative Adverserial Network we believed an Auto Encoder
would not have to generate an image for each category.

Not only that, but for this challenge it seemed like a novel approach.

An autoencoder is a neural network used to learn efficient codings of unlabeled data. 
The encoding is validated and refined by attempting to regenerate the input from the 
encoding. The autoencoder learns a representation (encoding) for a set of data, 
typically for dimensionality reduction, by training the network to ignore insignificant 
data (“noise”). 

![](visuals/AE.png)

We tried several different architectures using between 1 and 3 dense layers, 2 to 3 
convolutional layers and batch normalization.

For the best approach we created Autoencoder with convolution layers.
This is how it is regenerating images for orginal and anomalies.

![](visuals/ae_only.jpg)

metric|score
:--------------------------:|:--------------------------:
f1 | 0.21 
Accuracy | 0.53 
Precision | 0.12 
Recall | 0.83 

### AutoEncoder with classifier 

When we compared the results of the autoencoder, we noticed that if we could compare the 
reconstruction loss to the correct category the boundaries would more clearly defined and that,
if we could make a classification before sending the input to the AutoEncoder we would be able to 
use that specific threshold.

We then built a classification network for this purpose. So a sample is sent to both autoencoder 
and classification model. The classifier predicts a class and a pre-calculated threshold for 
the reconstruction loss is then given, with which the result of the autoencoder is then evaluated.

The architecture looked like this:

![](visuals/aec.jpg)

When comparing the reconstruction images to the input we get these. 
You can now more clearly see that there is compression and decompression from the artefacts of the 
image, but you can still see that for both normal and anomaly, it is very close to the original. 
The largest difference occurs on the last anomaly, where you can see that the scratches actually 
do not appear in the reconstructed images. 

![](visuals/ae+C.jpg)

metric|score
:--------------------------:|:--------------------------:
f1 | 0.46 
Accuracy | 0.96 
Precision | 0.57 
Recall | 0.39 

### Variational Auto Encoder

### Reasoning

For all the experiments done so far using AutoEncoders, no instances were found where the model
was actually learning. There are several possible reasons for this that we could ascertain.

Consider that the autoencoder accepts input, compresses it, and then recreates the original 
input. This is an unsupervised technique because all you need is the original data, without 
any labels of known, correct results. No matter how small the bottleneck the autoencoder seemed
to never have any trouble rebuilding the images provided from the latent representation.

We personally believe the most feasible reason is because the dataset is simply not noisy enough.
If there is no noise everything becomes signal and the model can never learn an underlying pattern.

That said the two main uses of an autoencoder are to compress data to two (or three) 
dimensions so it can be graphed, and to compress and decompress images or documents, which 
removes noise in the data. 

Allthough very useful this was not the usecase we were trying to present. Enter: 

### Variational Auto Encoders.

The difference is that a VAE assumes that the source data has some sort of underlying probability 
distribution (such as Gaussian) and then attempts to find the parameters of the distribution. 

![](visuals/VAE.png)

You can see that in the VAE plots the data using mean and standard deviation onto a distribution
in the probabilistic encoder part of the model and tries to find patterns in this latent space.

For our purpose the VAE finally gave some good results, using even the most basic model.

![](visuals/VAE-normal.png)

When testing on only normal samples the VAE generates a very similar result, and we can even see
in some of the lighter circles that it is aware of the place the circles are usually placed on, each 
dice. 

But it is not until we test on the abnormal samples that we can see the VAE in it's full potential.

![](visuals/VAE-anom.png)

The model tries to generate a similar image and it overlays a number of the normal samples over each
other, but can't seem to succeed. The distance between the abnormal and result increases and yields 
far better results. The final model was trained on 200 epochs, but has not been optimized since, so 
there is still a lot of room for improvement.

metric|score
:--------------------------:|:--------------------------:
f1|0.75
Accuracy|0.96
Precision|0.89
Recall|0.65

## Installation

- Clone this repository into your local environment with below command-

  `git clone https://github.com/ujjwalk00/anomalous_dice.git`

- Create python virtual environment

- Install all the required library with below command

  `pip install -r requirements.txt`
  
 ## Tech
 
 The following libraries were used:
 - **TensorFlow** For all things deep learing
 - **OpenCV** and **PIL** for image handling and processing
 - **sklearn** for calculating metrics
 - **NumPy** for fun

## Collaborators

([Ujjwal Kandel](https://github.com/UjjwalKandel2000), [Reena Koshta](https://github.com/reenakoshta10), and [Aubin](https://github.com/manwithplan))
