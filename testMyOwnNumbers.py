# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:43:30 2021

@author: lenna
"""

import numpy
import imageio
import matplotlib.pyplot
from neuralNetwork import NeuralNetwork

# set up network architecture
input_nodes = 784
hidden_nodes = 300
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.loadWeights("C:/Users/lenna/git/neuralnetworks_oreilly/"
              "Pretrained Network Weights/wih_1.csv", "C:/Users/lenna/git/"
              "neuralnetworks_oreilly/Pretrained Network Weights/who_1.csv")

img_array = imageio.imread("C:/Users/lenna/git/neuralnetworks_oreilly/"
                           "MyOwnNumbers/MyOwnNumber_7_2.png", as_gray=True)

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
# matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys',
#                          interpolation='None')
outputs = n.query(img_data)
label = numpy.argmax(outputs)
