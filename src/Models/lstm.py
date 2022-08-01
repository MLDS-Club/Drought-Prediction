# Standard libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the Keras libraries and packages
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Get model
def getModel(X_train):
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Droupout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = len(X_train[0])))

    model.compile(optimizer = "adam", loss = 'mean_squared_error')

    return model

