# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:42:04 2020

@author: csukh
"""

from layer import *

class NeuralNetwork():
    def __init__(self,input_dim,num_hl,nodes_per_hl,output_dim):
        """
        Initializes a neural network for classification; outputs are the one-hot
        encoded class labels
        
        params: 
            input_dim - number of features per input data point
            num_hl - number of hidden layers in the network
            nodes_per_hl - number of neurons in each of the hidden layers
            output_dim - nuymber of output neurons in the last layer (should equalt the number of classes)
        """
        
        self.input_dimension = input_dim
        self.inputs = np.zeros(input_dim)
        self.layers = list()
        
        # Construct the hidden layers
        num_nodes_pl = input_dim
        for c_layer in np.arange(num_hl):
            layer_name = 'hidden_layer_{}'.format(c_layer)
            num_nodes = nodes_per_hl[c_layer]
            temp_layer = Layer(layer_name,num_nodes,num_nodes_pl,'hidden')
            self.layers.append(temp_layer) 
            num_nodes_pl = num_nodes
        
        #Construct the output layer
        o_layer = Layer('output_layer',output_dim,num_nodes_pl,'output')
        self.layers.append(o_layer)
        self.output = np.zeros(output_dim)
        
        
    
    def feed_forward(self,inputs):
        """
        Implements forward propagation for the Neural Net. iterates through layers
        and looks at the output of the previous layer and takes the dot product 
        with the weights in each neuron of the current layer and takes sigmoid 
        of the result
        
        params: 
            inputs (int list): values to do forward propagation on.
        
        TODO: add ability to specify dropout to prevent overfitting and allow
            user to specify activation function per layer
        """
        
        self.inputs = inputs
        input_array = inputs
        
        for layer_num,current_layer in enumerate(self.layers):
            
            for ii,current_neuron in enumerate(current_layer.neurons):
                activation = np.dot(input_array,current_neuron.weights) + current_neuron.bias
                current_neuron.activation = activation
                neuron_output = current_neuron.sigmoid(activation)
                
                self.layers[layer_num].neurons[ii].output = neuron_output 
                self.layers[layer_num].neurons[ii].activation = activation
                self.layers[layer_num].output[ii] = neuron_output
                
            # Ouput array of the current layer becomes input to the next layer
            input_array = self.layers[layer_num].output
            
        #output of the last layer becomes output of the network
        self.output = input_array
        
    
    
    def backpropagation(self,targets,learning_rate):
        """
        performs the backpropagation step of training the neural net and generates 
        update and weight vectors for each neuron in each layer.
        
        params: 
            targets - one hot encoded vector corresponding to the desired class label
            learning_rate - specifies how aggresively you want to change the 
                weights/biases on each forward propagation step.
        """
        #Compute output layer gradients        
        layer_index = len(self.layers)-1
        
        while layer_index >= 0:
            
            c_layer = self.layers[layer_index]
            if c_layer.name == 'output_layer':
                for ii,c_neuron in enumerate(c_layer.neurons):
                    dE_do = targets[ii] - c_neuron.output
                    do_di = c_neuron.sigmoid_prime(c_neuron.activation)
                    di_dw = self.layers[layer_index-1].output
                    
                    self.layers[layer_index].neurons[ii].delta = dE_do*do_di
                    self.layers[layer_index].neurons[ii].biasUpdate = dE_do*do_di
                    self.layers[layer_index].neurons[ii].weightUpdates = learning_rate*di_dw*do_di*dE_do
            else:
                for ii,c_neuron in enumerate(c_layer.neurons):
                    
                    sigma_prime = c_neuron.sigmoid_prime(c_neuron.activation)
                    
                    #calculate the weights from the next layer and get the 
                    # deltas of next layer neurons
                    nl_weights  = list()
                    nl_deltas  = list()
                    for c_neuron in  self.layers[layer_index+1].neurons:
                        nl_weights.append(float(c_neuron.weights[ii]))
                        nl_deltas.append(float(c_neuron.delta))

                    delta = sigma_prime*np.dot(nl_deltas,nl_weights)
                    
                    # If we're at the last hidden layer, the previous layer is the inputs.
                    if (layer_index-1) < 0:
                        di_dw = self.inputs
                    else:
                        di_dw = self.layers[layer_index-1].output
                    
                    self.layers[layer_index].neurons[ii].delta = delta 
                    self.layers[layer_index].neurons[ii].biasUpdate = delta
                    self.layers[layer_index].neurons[ii].weightUpdates = learning_rate*delta*di_dw
                    
            layer_index -= 1
        
    
    def update_weights(self):
        """
        Change the weight vector for each neuron by adding the weight updates
        we calculated in the backpropagation step
         
        params: none
        """        
        for layer_idx,layer in enumerate(self.layers):
            for neuron_idx,c_neuron in enumerate(layer.neurons):
                
                updates_vec = self.layers[layer_idx].neurons[neuron_idx].weightUpdates
                self.layers[layer_idx].neurons[neuron_idx].weights += updates_vec 
                bias_update = self.layers[layer_idx].neurons[neuron_idx].biasUpdate
                self.layers[layer_idx].neurons[neuron_idx].bias += bias_update 
    
    def train(self,inputs,targets,learning_rate):
        """
        Perform a full training iteration consisting of a feed forward step, 
        back propagation and  updating the weights in the network
        
        params:
            inputs - training exemplar
            targets - one hot encoded class corresponding to the input
            learning_rate - specifies how aggresively you want to update the 
                weights after each backprobagation step.
        """
        self.feed_forward(inputs)
        self.backpropagation(targets,learning_rate)
        self.update_weights()
        
    def print_error(self,targets):
        """
        prints the difference between the target vector and the neural net's output
        The number this function prints out is the MSE (Mean Squared Error).
        
        """
        error = np.sum(np.subtract(targets,self.output)**2)
        print(error)
        
        
        
        
        
        
        
        
        
    