## ANN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing the liberary which is responsible for creating ANN

import keras
from keras.models import Sequential # It is responsible for creating ANN, CNN 
from keras.layers import Dense # Hidden layer
from keras.layers import LeakyReLU, PReLU, ELU # activation function
from keras.layers import Dropout # It is regularization paramenter



# Reading the data
df = pd.read_csv('G:\Deep_learning_coding\pratice_1\Churn_Modelling.csv')
print(df)

# Data seperation 
x = df.iloc[:, 3:13]
print('The x data is \n', x.head())

y = df.iloc[:, 13]
print('The y data is \n', y.head())

# checking the null values 

print('The null value for x column is \n', x.isnull().sum())
print('The null value for y column is \n', y.isnull().sum())

# checking the columns name 
print(x.columns)

# creating a dummy variables for categorical variable 
geography = pd.get_dummies(x['Geography'], drop_first = True)
print(geography)
gender = pd.get_dummies(x['Gender'], drop_first = True)
print(gender)

# Concatenate the data frame
x = pd.concat([x, geography, gender], axis = 1)
print("The x data is \n",x)

# Droping the column 
x = x.drop(['Geography', 'Gender'], axis = 1)
print("The x final data is \n",x)

# splitting the data set
x_train, x_test , y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=0)

# scaling the x_train, x_test
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Initialization Sequential liberary
classifier = Sequential() # this will be an empty neural network currently 

# Adding the first input layer and the first hidden layer 

classifier.add(Dense(6, kernel_initializer='he_uniform', activation='relu', input_shape=(11,)))


