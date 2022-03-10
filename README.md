# digits_classification

The project aims to Train a simple Neural Network clssifier to classify the different handwritten digits (labels
between zero and nine)

After been trained, the proposed model will be evaluaded on an unseed
 data based on the accuracy and the confusion matrix
 
At the end some misclassified images will be ploted  for mistake analysis

Requirements:
-	Conda or Miniconda [optional] : for simple dependancies management
-	cuda and cudnn [optional] : if you have a GPU and the corresponding driver installed
-	Keras
-	Tensorflow 

installation :

# create and activate the working environment
-	create -n mnist_env python=3.7
-	conda activate mnist_env

# install cuda and cudnn (the newest version (cuda 11.3.1 and cudnn 8.2.1) at the time is used in this project)
-	conda install cudnn

# install keras and tensorflow (the newest version(2.6.0) at the time is used in this project)
-	conda install tensorflow-gpu
-	pip install keras==2.6.0
-	pip install tensorflow-addons
