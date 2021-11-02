# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:16:23 2021

@author: lenna
"""

from neuralNetwork import NeuralNetwork
import numpy
from datetime import datetime

# set up network architecture
input_nodes = 784
hidden_nodes = 300
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load MNIST training data CSV file into a list
folder = "C:/Users/lenna/git/neuralnetworks_oreilly/MNIST Dataset/"
file = "mnist_train.csv"
path = folder + file
training_data_file = open(path, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the network

# introduce number of epochs (how often training data is used for training)
epochs = 5

# go through all records in training_data_list (epochs times)
start_time = datetime.now()  # we want to know how long training took

for e in range(epochs):
    for record in training_data_list:
        # split the recors by the commas
        all_values = record.split(',')
        # scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create target output values
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

end_time = datetime.now()
print("training took: ", end_time - start_time)

# load MNIST test data CSV file into a list
folder = "C:/Users/lenna/git/neuralnetworks_oreilly/MNIST Dataset/"
file = "mnist_test.csv"
path = folder + file
test_data_file = open(path, 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the network

# scorecard for identifying how well the nn works; initially empty
scorecard = []

# go through all records in test_data_list
for record in test_data_list:
    # split the recors by the commas
    all_values = record.split(',')
    # correct answer is the first value
    correct_label = int(all_values[0])
    # print("correct label: ", correct_label)
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # print("network's answer: ", label)
    # append correct or incorrect to scorecard
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

    pass

# evaluate the scorecard
print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
