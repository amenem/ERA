# Session 10 Assignment

This is a modular implementation for S9 - Assignment QnA.

Contents: 
1. custom_resnet.py
3. dataset.py
3. build_dataloader.py
4. trainer.py
5. S10.ipynb
6. README.md

## custom_resnet.py
In this file , custom resnet model network is defined.

## dataset.py
It has the Custom dataset class to read cifar10 data which expect albumentation transforms. 

## build_dataloader.py
This module contains methods to get train loader and test loader. It also has the augmentations for the datasets.

## trainer.py
It trains the model on the input data, and evaluate each epoch on the test data

## S10.ipynb
Notebook to get the train/test data loaders, create model instance, optimizer. And train it.


