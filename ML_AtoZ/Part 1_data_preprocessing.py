"""
Machine Learning A-Z: Hands-On Python & R In Data Science
created by Kirill Eremenko, Hadelin de Ponteves, SuperDataScience Team
https://www.udemy.com/machinelearning/

Section 1: Data Preprocessing Template
"""
#Data Preprocessing
#type CMND+I to inspect (or 'learn more')
#to comment a section, use 3 quation """ """

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset, you must set a working directory
dataset = pd.read_csv('Data.csv')

#lat's take a lok at the dataset
dataset.head()
dataset.info()

#let's create matrix of features
x = dataset.iloc[:, :-1].values
print(x)

#create the dependent variable vector (y-value)
y = dataset.iloc[:,3:4].values
print(y)

#how do we handle missing data?
#1. remove observation? = no, too risky
#2. what about replacing missing data with the mean of the column

#import preprocessing Imputer class from Scik-it learn
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

#we will only fit it into the columns with the missing data (col 1, 2)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x) #missing data has been replaced by the mean


#Encoding categorical data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:,0])


#note, machine learning is based on equation
#Encoding the dependent Variable
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split data into training and test set
##from sklearn.cross_validation import train_test_split  - old
#import the library required
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 0) 


#future scaling: put all variables on the same scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Do we need to fit and transform the dummy variable? It seems to be already scaled
#answer: it depends on the context and how you want to keep the interpretation in the model


