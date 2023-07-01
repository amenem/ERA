# Session 9 Assignment

This is a modular implementation for S9 - Assignment QnA.

Contents: 
1. model/model1.py
2. model/model2.py
3. dataset.py
4. build_dataloader.py
5. trainer.py
6. S9.ipynb
7. README.md

## model/model1.py
In this file , model network is defined.
Three Convolutional blocs With 4 conv layer each with BatchNorm and Relu. The last layer is strided convolution instead of maxpool. Receptive field of 49 is reached at the last layer of the last bloc. Dropout is not used as train acc is already lagging behind the test accuracy, due to augmentations.

## model/model2.py
In this file , model2 network is defined.
It has the same network skeleton except the last layer in the blocs, instead of strided conv layer, dilated convolution is used.

## dataset.py
It has the Custom dataset class to read cifar10 data which expect albumentation transforms. 

## build_dataloader.py
This module contains methods to get train loader and test loader. It also has the augmentations for the datasets.

## trainer.py
It trains the model on the input data, and evaluate each epoch on the test data

## S9.ipynb
Notebook to get the train/test data loaders, create model instance, optimizer. And train it.


