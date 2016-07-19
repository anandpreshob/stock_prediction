# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:19:32 2016

@author: anandpreshob
This code is for stock prediction one machine learning challenge
Given a dataset that consists of the (simulated) daily open­to­close changes of a set of 10
stocks: S1, S2, …, S10. Stock S1 trades in the U.S. as part of the S&P 500 Index, while stocks S2, S3, …,
S10 trade in Japan, as part of the Nikkei Index.
Your task is to build a model to forecast S1, as a function of S1, S2, …, S10​. You should build your
model using the first 50 rows of the dataset. The remaining 50 rows of the dataset have values for S1
missing: you will use your model to fill these in.

"""
#%% Import libraries
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
#%% Read data and preprocessing
data=pd.read_csv('/media/anandpreshob/Anand/Anand/MLCV/kaggle/correlation_one/stock_returns_base150.csv')
y_train=np.asarray(data['S1'][0:50])
x=data[['S2','S3','S4','S5','S6','S7','S10']]
xdata=x.as_matrix()
x_train=xdata[0:50,:]
x_test=xdata[50:100,:]

#%% Cross validation to evaluate the classifier
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(x_train,y_train, test_size=0.4, random_state=0)
ranf = RandomForestRegressor()
ranf.fit(X_train, Y_train)
print(ranf.score(X_val, Y_val))
# Fit the model using all the available data
ranf.fit(x_train, y_train)
ranf.predict(x_test)