# Neural network to recognize handwritten numbers
This program allows you to initialize, train your network and do the "honest recognition".
 
### Files
#### main.cpp 
Contains user menu.
#### betterthanmnist.h 
Reads pixels of image from file and the correct answer. Each pixel is a float number.
#### neuralnetwork.h 
Neural network class. Does recognition and backward pass.
#### timer.h 
Calculates time.
#### honestTest.txt
Contains pixels, but does not contain the answer. The neural network can't check itself, only try to guess the number.
#### init90.txt
Contains layer weights. You may initialize neural network from this file. It has about 90% accuracy.
#### learningData.txt
62MB file. Contains data to train and test neural network.
