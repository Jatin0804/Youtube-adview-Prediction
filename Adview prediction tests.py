#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd


# In[53]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[54]:


# read the training data
data_train = pd.read_csv("train.csv")


# In[55]:


# check the head of data
data_train.head()


# In[56]:


data_train.tail()


# In[57]:


# check the shape of dataset
data_train.shape


# In[58]:


# map each category of video to a number
category = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
}


# In[59]:


data_train["category"] = data_train["category"].map(category)


# In[60]:


# check again the data
data_train.head()


# In[61]:


# remove character 'F' present in data
data_train = data_train[data_train.views != 'F']
data_train = data_train[data_train.likes != 'F']
data_train = data_train[data_train.dislikes != 'F']
data_train = data_train[data_train.comment != 'F']


# In[62]:


# convert values to integers
data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])


# In[63]:


column_vidid = data_train["vidid"]


# In[64]:


# sklearn library
from sklearn.preprocessing import LabelEncoder


# In[65]:


# encode features
data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])


# In[66]:


data_train.head()


# In[67]:


# import time 
import datetime
import time


# In[68]:


# convert time_in_sec for duration
def check(x):
    year = x[2:]
    hours = ''
    minutes = ''
    seconds = ''
    mm = ''
    P = ["H", "M", "S"]
    for i in year:
        if i not in P:
            mm += i
        else:
            if i == "H":
                hours = mm
                mm = ''
            elif i == "M":
                minutes = mm
                mm = ''
            else:
                seconds = mm
                mm = ''
    if hours == '':
        hours = '00'
    if minutes == '':
        minutes = '00'
    if seconds == '':
        seconds = '00'
    bp = hours + ":" + minutes + ":" + seconds
    return bp


# In[69]:


train = pd.read_csv("train.csv")


# In[70]:


mp = pd.read_csv("train.csv")["duration"]


# In[71]:


time = mp.apply(check)


# In[72]:


def func_sec(time_string):
    hours, minutes, seconds = time_string.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


# In[73]:


time1 = time.apply(func_sec)


# In[74]:


data_train["duration"] = time1


# In[75]:


data_train.head()


# In[76]:


# visulaization


# In[77]:


plt.hist(data_train["category"])
plt.show()


# In[78]:


plt.plot(data_train["adview"])
plt.show()


# In[79]:


# remove videos with adview greater than 2000000
data_train = data_train[data_train["adview"] < 2000000]


# In[80]:


# heatmap
import seaborn as sns


# In[81]:


f, ax = plt.subplots(figsize = (10, 8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot=True)
plt.show()


# In[82]:


# split the data
Y_train = pd.DataFrame(data = data_train["adview"].values, columns = ['target'])
data_train = data_train.drop(["adview"], axis = 1)
data_train = data_train.drop(['vidid'], axis = 1)


# In[83]:


data_train.head()


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state=42)


# In[86]:


X_train.shape


# In[87]:


# Normalize the data
from sklearn.preprocessing import MinMaxScaler


# In[88]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[90]:


X_train.mean()


# In[89]:


# Evaluation metrics
from sklearn import metrics


# In[91]:


def print_error(X_test, y_test, model_name):
    predictions = model_name.predict(X_test)
    print("Mean absolute error: ", metrics.mean_absolute_error(y_test, predictions))
    print("Mean squared error: ", metrics.mean_squared_error(y_test, predictions))
    print("Root Mean Squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[92]:


# Linear Regression
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test, y_test, linear_regression)


# In[93]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test, y_test, decision_tree)


# In[95]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators,max_depth=max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
random_forest.fit(X_train, y_train)
print_error(X_test, y_test, random_forest)


# In[96]:


# Support vector regressor
from sklearn.svm import SVR
support_vectors = SVR()
support_vectors.fit(X_train, y_train)
print_error(X_test, y_test, support_vectors)


# In[99]:


# Artificial neural Network
import keras
from keras.layers import Dense


# In[100]:


ann = keras.models.Sequential([
    Dense(6, activation="relu", input_shape=X_train.shape[1:]),
    Dense(6, activation="relu"),
    Dense(1)
])


# In[101]:


optimizer = keras.optimizers.Adam()
loss = keras.losses.mean_squared_error


# In[102]:


ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])


# In[104]:


print(type(X_train))
print(type(y_train))


# In[103]:


history = ann.fit(X_train, y_train, epochs=100)


# In[105]:


x_train = np.array(X_train)
y1_train = np.array(y_train)


# In[109]:


print(type(X_train))
print(type(y1_train))


# In[112]:


history = ann.fit(X_train, y1_train, epochs=100)


# In[118]:


print(X_train.shape)
print(y_train.shape)


# In[117]:


print(np.isnan(y_train).sum())
print(y_train.dtype)


# In[114]:


y_train = np.array(y_train)


# In[116]:


history = ann.fit(X_train, y_train, epochs=100)

# In[120]:


# saving Scikit-learn models
import joblib


# In[121]:


joblib.dump(decision_tree, "decision_tree_yt_adview.pkl")


# In[123]:


# saving keras ANN
ann.save("ann_yt_adview.h5")


# In[ ]:




