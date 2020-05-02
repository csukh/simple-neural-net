# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:08:34 2020

@author: csukh
"""

from neural_network import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Initialize Neural Net
wine_net = NeuralNetwork(13,2,[5,3],3)

#Load dataset and break it into training and test sets
wine_data = pd.read_csv(r'wine.csv')
cols = wine_data.columns
y = wine_data['label']

X = wine_data[cols[1:]]
rs = RobustScaler()
X_scaled = rs.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size = 0.3, train_size = 0.7)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#Make 50 passes throug the training set (50 epochs)
for i in np.arange(200):
    
    for row in range(len(X_train)):
        
        if y_train.iloc[row] == 1:
            tgt = [0,0,1]
        elif y_train.iloc[row] == 2:
            tgt = [0,1,0]
        else:
            tgt = [1,0,0]
        
        train_data = X_train.iloc[row].tolist()
        wine_net.train(train_data,tgt,0.05)
        
        wine_net.print_error(tgt)
        
        
## Testing
errors_vec = list()
for row in range(len(X_test)):
    
    if y_test.iloc[row] == 1:
        tgt = [0,0,1]
    elif y_test.iloc[row] == 2:
        tgt = [0,1,0]
    else:
        tgt = [1,0,0]
        
    test_sample = X_test.iloc[row].tolist()
    
    wine_net.feed_forward(test_sample)
    predicted = np.round(wine_net.output)
    errors_vec.append(np.sum(np.abs(np.subtract(tgt,predicted))))
    

# Get average error
print(np.mean(errors_vec))