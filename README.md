DeepLearningToolBox
===================
This is a Deep Learning toolbox implemented by C/C++

You need to install Eigen Matrix Libray and OpenCV 2.4.3 (or other alternative) in order to build this project.

Some datasets are in zip file,you need to unzip them and copy to correct folders.

You need to download the mnist dataset from http://yann.lecun.com/exdb/mnist/ and unzip to the correct folder,make sure it has the same name to what is referenced in the code,please rename it if not.

The getConfigXXValues functions may not work due to the end of line format of "ParamConfig.ini" file,you should convert it to the windows platform end of line format.

The Deep Neural Network implemented by stacked autoencoders with proper paramters has a classification accuracy of 98% or even better on the mnist dataset(trained on training set and tested on test set).


