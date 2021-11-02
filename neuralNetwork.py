# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:35:06 2021

@author: lenna
"""

# imports
import numpy
import scipy.special  # needed for sigmoid function


# neural network class definition
class NeuralNetwork:

    # initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # number of nodes in each of the three layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # link weight matrices
        # wih (input --> hidden) and who (hidden --> output)
        # weights inside the arrays are w_i_j, where link is from node i to
        # node j in the next layer
        # wih: hiddennodes x inputnodes
        # who: outputnodes x hiddennodes
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                        (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                        (self.onodes, self.hnodes)))

        # activation function
        # (currently the sigmoid function from scipy.special)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert lists to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # errors
        output_errors = targets - final_outputs

        # hidden layer errors = output errors split by weights
        # and recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weights between hidden and output layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs
                                         * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update weights between input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs
                                         * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    # load pretrained network
    def loadWeights(self, file_wih, file_who):
        self.wih = numpy.loadtxt(file_wih, delimiter=',')
        self.who = numpy.loadtxt(file_who, delimiter=',')

        self.inodes = numpy.shape(self.wih)[1]
        self.hnodes = numpy.shape(self.wih)[0]
        self.onodes = numpy.shape(self.who)[0]
        pass
