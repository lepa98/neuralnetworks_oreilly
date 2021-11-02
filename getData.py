# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:06:46 2021

@author: lenna
"""

# imports
import numpy
import matplotlib.pyplot

# get data from file
folder = "C:/Users/lenna/git/neuralnetworks_oreilly/MNIST Dataset/"
file = "mnist_train_100.csv"
path = folder + file
data_file = open(path, 'r')
data_list = data_file.readlines()
data_file.close()

# plot first entry of file as image
all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

# scale input to range 0.01 to 1.0
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

# generate target values
onodes = 10  # there are 10 digits 0 ... 9 which have to be represented
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
