# digits_classification

The project aims to Train a simple Neural Network clssifier to classify the different handwritten digits (labels
between zero and nine)

After been trained, the proposed model will be evaluated on an unseed
 data based on the accuracy and the confusion matrix
 
At the end some misclassified images will be ploted  for mistake analysis

# Requirements:
-	python3
-	Conda or Miniconda [optional] : for simple dependancies management
-	cuda and cudnn [optional] : if you have a GPU and the corresponding driver installed
-	Keras
-	Tensorflow 
-	matplotlib
-	seaborn
-	pydot

# installation :
##  create and activate the working environment:
-	create -n mnist_env python=3.7
-	conda activate mnist_env

##  install cuda, cudnn, pydot and tensorflow :the newest version (cuda_11.3.1, cudnn_8.2.1 and tensorflow_2,6) at the time is used in this project
-	conda install cudnn
-	conda install tensorflow-gpu
-	conda install pydot

##  install pip libraries:
-	pip install keras==2.6.0
-	pip install tensorflow-addons
-	pip install matplotlib

##  clone the project: 
- you can use TortoiseGit(simple) or git the git command 


# Training and evaluation :
##  dataset:
The dataset used is the MNIST dataset provided by tensorflow (automaticaly douwnloaded).
It consists of 28x28 grayscale images of the 10 digits (0,1,...,9) with the corresponding labels and 
divided into 60,000 samples(images,labels) for training and 10,000 samples for testing

To train and evaluate the model, just run the "trainer.py" file.
The default training epoch is 30, after what the trained model is saved by default in 
"trained_model/model.h5"
Note : The model architecture , the confusion matrix, the training evolution curves and
a picture showing some missclassified images are stored by default in trained_model folder too.

# Prediction :
To make a prediction, the function predict in "predictor.py" file must be applied on 
the data.