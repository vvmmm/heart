#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading the csv data to a pandas dataframe
heart_data=pd.read_csv("heart disease prediction last.csv")


# In[3]:


# print first five raws of dataset
heart_data.head()


# In[4]:


#print last five raws of the dataset
heart_data.tail()


# In[5]:


#to check no. of raws and columns in dataset
heart_data.shape


# In[6]:


# to get some information about the dataset
heart_data.info()


# In[7]:


# to chech missing values in dataset
heart_data.isnull().sum()


# In[8]:


#statistical measures about data
heart_data.describe()


# In[9]:


# to checking the distribution of target variable
heart_data['target'].value_counts()


# 1----> defective heart
# 
# 0----> healthy heart

# ### splitting the features and target heartdisease

# In[10]:


X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']


# In[11]:


print(X)


# In[12]:


print(Y)


# # splitting the data into training data and test data

# In[13]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[14]:


#check no. of training data and test data
print(X.shape,X_train.shape,X_test.shape)


# # model training 

# # logisticRegression  model for classification

# In[15]:


model = LogisticRegression() 


# In[16]:


model.fit(X_train,Y_train)


# In[17]:



heart_data 


# In[ ]:





# # model evaluation

# # Accuracy score as evaluation metrics

# In[18]:


#accuracy on train
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[19]:


print("accuracy on training data:",training_data_accuracy)


# In[20]:


#accuracy on train
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[21]:


print("accuracy on training data:",test_data_accuracy)


# # buildinig a predictive system

# In[22]:


input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)
#change input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)


# In[23]:


#reshape the numpy array as we are predicting for only one instance
input_data_reshaped =input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("the person does not have heart disease")
else:
    print("the person have heart disease")


# In[ ]:




