# logo-recognition
Deep Learning project: fast food logo recognition for 5 companies using Python, Tensorflow and OpenCV. Got to 92% accuracy on test set.

resize-and-create.py contains the resizing of the original images for each company and the creation of samples based on each picture using opencv_createsamples
model.py loads the data, makes the CNN in tensorflow and trains the model. I used code from the 'Deep Learning by Google' Udacity course as my starting code to get familiar with Tensorflow
predict.py retores the weights of the trained model and uses it to predict a new image

More details in the aaai-paper.

