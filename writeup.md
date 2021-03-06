# **Traffic Sign Recognition** 

## Writeup
### Author : Mohan Barathi

### The code implemented in IPython notebook is much of self explanatory, with corresponding markdown cells explaining the code part.
### Hence, this writeup shall describe about how each and every [rubric points](https://review.udacity.com/#!/rubrics/481/view) are satisfied in the implementation.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/augment.png "Augmentation"
[image3]: ./examples/after_generation.png "After generation"
[image4]: ./examples/accuracy.png "Accuracy"
[image5]: ./examples/web_images.png "Web Images"
[image6]: ./examples/TrafficSignal.png "Traffic Signal"
[image7]: ./examples/softmax5.png "softmax5"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/feature_map.png "feature_map"
[image10]: ./examples/speed20.png "speed20"
[image11]: ./examples/softmax_output1.png "soft_max_1"
[image12]: ./examples/softmax_output2.png "soft_max_2"
[image13]: ./examples/softmax_output3.png "soft_max_3"
[image14]: ./examples/softmax_output4.png "soft_max_4"
[image15]: ./examples/softmax_output5.png "soft_max_5"
[image16]: ./examples/softmax_output6.png "soft_max_6"
[image17]: ./examples/softmax_output7.png "soft_max_7"
[image18]: ./examples/softmax_output8.png "soft_max_8"
[image19]: ./examples/softmax_output9.png "soft_max_9"
[image20]: ./examples/softmax_output10.png "soft_max_10"




## Rubric Points
---
# CATEGORY : Files Submitted
## CRITERIA : Submission Files

This project repository contains
* [Ipython notebook with code](./Traffic_Sign_Classifier.ipynb)
* [HTML output of the code](./Traffic_Sign_Classifier.html)
* A writeup report (markdown)


# CATEGORY : Data Set Summary & Exploration
## CRITERIA : Dataset Summary
Bar graph is used to calculate summary statistics of the traffic signs
> Code blocks 6, 7, and 8

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

## CRITERIA : Exploratory Visualization
> Bar graph, of the data extracted from signnames.csv file, as a pandas series.
![alt text][image1]

# CATEGORY : Design and Test a Model Architecture
## CRITERIA : Preprocessing
The images vary in brightness, with **should not** be fed to the CNN model. This will lead the model to account brighness also as one of the parameter.
**To overcome this, we can use [histogram normalisation](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) across all three channels.**
The image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data. 
> Normalizing the images, before augmenting them will lead to multiple copies of same image. This ultimately lead to overfitting of the model, producing 99% of accuracy in training set, but less than 80% of accuracy in validation set. Hence, normalization is moved after the augmentation step. 
![alt text][image2]
As the data set is small, the training of Model may easily result in overfitting. Also, our model might be **biased** towards the classes that have many samples in training set. 
> Here, I have created a set of aumented images, to overcome this problem
### Here, new set of augmented images are produced, using [affine transform](https://docs.opencv.org/3.4.3/d4/d61/tutorial_warp_affine.html)
> The size of Training set is increased 4 times, and the count of augmented images for each sample is based on their original count in training set
> This step takes more time than the training of neural netowrk. As parallelism cannot be exploited for producing augmented images, each and every image are generated sequentially, and appended to the training set. This is faster in bare metal, than on cloud workspace. Using the GPU instance that Udacity provides (for training the models) doesnot help for this purpose. This will exhaust the GPU limit that we have.

| **Workspace used**        	| **Time Taken**		| 
|:-----------------------------:|:---------------------:| 
| Bare metal (Laptop)       	| 5 hours(approx.)  	| 
| Udacity GPU instance      	| 6 hours(approx.)     	|
| Udacity Normal Cloud Workspace| 9 hours(approx.)		|

> So it's a good idea to generate these images, store the images in a pickle file (~400 mb), upload it to the udacity workspace and start the GPU instance.

After generation, the count of each class in the training set is equalised.
![alt text][image3]

## CRITERIA : Model Architecture
> The Lenet Architecture used in classroom for CNN, is used here. Few Dropouts are added for reducing the problem of overfitting

| Layer             |     Description	                          |
|:-----------------:|:-------------------------------------------:|
| Input         	| 32x32x3 RGB image					          |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU				| outputs 28x28x6							  |
| Max Pooling 2x2   | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU			    | outputs 10x10x16							  |
| Max Pooling 2x2   | 2x2 stride, valid padding, outputs 5x5x16   |
| Flatten       	| outputs 400                                 |
| Fully Connected   | outputs 120                                 |
| Drop out          | Keep probability = 0.7                      |
| Fully Connected   | outputs 84                                  |
| Drop out          | Keep probability = 0.7                      |
| Fully Connected   | outputs 43                                  |

To Train the model, the LeNet Architecture is used as the solid starting point. 
The LeNet Architecture is so good that the initial try gave 89% accuracy. However, 93% or above is considered as the pass criteria.

## CRITERIA : Model Training

**The initial attempts gave a fair good accuracy. But when augmented images are added to the training set, the training set accuracy reached 100%, but validation set accuracy decreased to 80%. A huge overfitting was identified.**

* later, upon changing the order of normalization and augmented image generation steps, this overfitting problem was solved.
* This was due to the fact that, if only affine transform is used, after applying all other preprocessing techniques, the same image with same properties might get generated multiple times. This will lead to overfitting of the model. So, preprocessing steps can be done after augmentation of training data.

Hyper parameters :

| Epochs     | 100  |
|:------------:|:-------|
| batch-size | 128  |
| Learning rate | 0.001 |

Adam Optimizer is used in this training, as it is more efficient that Gradient Descent Optimizer

## CRITERIA : Solution Approach

Various approaches were used to optimize the model, and the outcome of those experiments are discussed here.

Initially, the training set was fed to the LeNet Model, just after **normalizing it around mean 0.**
* training set accuracy : 93%
* validation set accuracy : 89%

As a step towards data peprocessing, **Histogram normalization** was applied over all 3 R,G abd B channels, to normalize the brightness of all the images.
* training set accuracy : 98%
* validation set accuracy : 91%

The Model started to overfit the training data. One reason could be because of small size of the dataset, and the huge difference in counts of samples available under each classes. The model might bias towards the classes with more number of samples.
So, **augmented images were added using affine transform**, which increased the size of training set 4 times.

Again, the accuracy was 
* training set accuracy : 98%
* validation set accuracy : 90%

To overcome this problem, **2 dropout layers were added to the Lenet Architecture**, just after the 2 fully connected layers.
These droupouts will help the model not to be too dependent on the training data set, and can generalize.

Now the accuracy was 
* training set accuracy : 98%
* validation set accuracy : 93%

Changing the Learning rate, epoch, batch size, adding more droupout layers and changing the keep probability of droupouts actually decresed the overall accuracy.

* training set accuracy : 85%
* validation set accuracy : 70%

So, these changes were reverted back.

Normalising the data and producing huge set of augmented images, just added copies of same image (with slight changes), multiple times. Also, the newly added pixels as a part of augmentation, were not normalized.

Hence, the Normalization step was moved after the Augmentation of training set.

**Finally, the model results were:**
* **training set accuracy : 99.5**
* **validation set accuracy : 97.6**
* **test set accuracy : 94.5**

![alt text][image4]
 

# CATEGORY : Test a Model on New Images
## CRITERIA : Acquiring New Images

The five German traffic signs that I found on the web:

![alt text][image5]

> The performance of these new 5 images might be poor than the test or training set images.  This is mainly because, the signs are more or less similar to the signs of predicted outcome.

> **Genaral Caution** looks similar to **Traffic signal**

> **Go straight or right** looks similar to **no entry**

## CRITERIA : Performance on New Images

As expected, the performace is poor, and only few images were correctly identified.

![alt text][image6]

![alt text][image7]

**Here are the results of the prediction:**

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing      		| Wild animals crossing   									| 
| Go straight or right     			| No entry 										|
| End of no passing					| End of no passing											|
| Bumpy Road	      		| Bumpy Road					 				|
| Traffic signal			| General Caution      							|


The model was able to correctly guess 3 of the 5 traffic signs.
> As we can see, the performance on these 5 images is **only 60%** , but the performance of our model in **training set is 99.5**, **validation set is 97.6**, and **testing set is 94.5**

> This is mainly because, the signs with incorrect predictions look are more or less similar to the signs of predicted outcome.

> **Genaral Caution** looks similar to **Traffic signal**

> **Go straight or right** looks similar to **no entry**

This can be solved by using a **generating a better and huge augmented dataset for training the model.**

## CRITERIA : Model Certainty - Softmax Probabilities
> The first five top probabilities of classes are deduced in the code block 26, under the VISUALIZATIONS OF THE SOFTMAX PROBABILITIES category.

Here, the probabilities are presented in form of bar graphs.

---

![alt text][image11]
![alt text][image12]

---

![alt text][image13]
![alt text][image14]

---

![alt text][image15]
![alt text][image16]

---

![alt text][image17]
![alt text][image18]

---

![alt text][image19]
![alt text][image20]



# Optional Category :  Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
* The First 2 layers' operations are taken out from Neural network, and are fed to the provided function, along with normalised sample image to visualize the Nural network operation. (Last block of IPython notebook.)

![alt text][image10]
![alt text][image9]
