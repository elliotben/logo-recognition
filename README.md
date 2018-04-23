# logo-recognition
Fast food logo recognition for 5 companies using Python, Tensorflow and OpenCV using a Convolutional Neural Network. Got to 92% accuracy on test set.

resize-and-create.py contains the resizing of the original images for each company and the creation of samples based on each picture using opencv_createsamples
model.py loads the data, makes the CNN in tensorflow and trains the model
predict.py retores the weights of the trained model and uses it to predict a new image

More details in the aaai-paper.

