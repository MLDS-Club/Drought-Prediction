import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sys

sys.path.insert(0, '/Users/arshianayebnazar/Documents/GitHub/DroughtPrediction/src/')
print("Path: ", sys.path)

from Models.lstm import getModel

df = pd.read_csv('data/cleandrought2010-2022')

X_train = []
y_train = []

for i in range(52, len(df)):
    X_train.append(df[i-52:i, :])
    y_train.append(df[i, :])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = getModel(X_train)

model.fit(X_train, y_train, epochs=1, batch_size = 32)