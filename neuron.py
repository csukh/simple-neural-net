# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:28:41 2020

@author: csukh
"""

import numpy as np

class Neuron:
    def __init__(self, name, weights_length):
        self.name = name
        self.weights = np.random.randint(-100,high=100,size=weights_length)/100
        self.activation = 0
        self.delta=0
        self.bias = np.random.randint(-100,high=100,size=1)/100
        self.output = 0
        self.weightUpdates = np.zeros(self.weights.shape)
        self.biasUpdate = 0
        
    def sigmoid(self, x):
        # Sigmoid function
        return 1.0/(1.0 + np.exp(-1.0*x))

    def sigmoid_prime(self, x):
        # sigmoid derivative
        return self.sigmoid(x)*(1.0 - self.sigmoid(x))