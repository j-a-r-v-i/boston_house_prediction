# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:15:14 2018

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
X_opt=X[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

#splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train.reshape(-1,1))


#fitting the svm model to dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X_train,y_train)

#predicting the value
y_pred=sc_y.inverse_transform(regressor.predict(X_test))
plt.scatter(y_test,y_pred)
print(sklearn.metrics.mean_squared_error(y_test,y_pred))
