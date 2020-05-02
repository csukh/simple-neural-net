# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:33:31 2020

@author: csukh
"""

from neuron import *

class Layer():
    def __init__(self,name,num_neurons,previous_layer_neurons,layerType):
        
        self.previous_layer_neurons = previous_layer_neurons
        self.name = name
        self.output = np.zeros(num_neurons)
        
        self.neurons = list()
        for c_neuron in np.arange(num_neurons):
            neuron_name = self.name + '_neuron_{}'.format(c_neuron)
            number_of_weights = previous_layer_neurons
            neuron = Neuron(neuron_name, previous_layer_neurons)
            self.neurons.append(neuron)