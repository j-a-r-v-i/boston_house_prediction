# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:36:17 2018

@author: archit bansal
"""

# boston housing prices

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Importing the dataset
dataset = pd.read_excel('boston.xls')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,13].values

#building th optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)

#applying randomforestRegression on training data
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(X_train,y_train)

#predicting the result
y_pred=regressor.predict(X_test)
plt.scatter(y_test,y_pred)
 #meansquarevalue
print(sklearn.metrics.mean_squared_error(y_test,y_pred))