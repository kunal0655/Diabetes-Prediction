#!/usr/bin/env python
# coding: utf-8

# In[90]:


#Import Dependencies :
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[11]:


#Loading the diabetes dataset into a pandas DataFrame
df=pd.read_csv('diabetes.csv')
df.head()


# In[16]:


#Number of rows and columns in the Data set:
df.shape


# In[17]:


#Getting  the statistical measures of the data


# In[18]:


df.describe()


# In[23]:


# We will check the total diabetic patient and non diabetic patient in dataset
#0-> Non Diabetic
#1-> Non Diabetic
df['Outcome'].value_counts()


# In[24]:


df.groupby('Outcome').mean()


# In[26]:


#Separating the data and labels
X=df.drop(columns='Outcome', axis=1)


# In[27]:


Y=df.Outcome


# In[29]:


print(X)


# In[30]:


print(Y)


# In[31]:


#Data Standarization


# In[85]:


SS=StandardScaler()


# In[98]:


SS.fit(X)


# In[34]:


standarized_data=SS.transform(X)


# In[35]:


print(standarized_data)


# In[39]:


X=standarized_data
Y=df.Outcome


# In[41]:


X


# In[42]:


Y


# In[48]:


#Train Test and Split
#Some classification problems do not have a balanced number of examples for each class label. 
#As such, it is desirable to split the dataset into train and test sets in a way that preserves the same proportions 
#of examples in each class as observed in the original dataset. 
#This is called a stratified train-test split.
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20, stratify=Y)


# In[50]:


# We can check the data used in training and testing
print(X.shape, X_train.shape, X_test.shape)


# In[56]:


from sklearn.linear_model import LogisticRegression


# In[57]:


#Assigning the model
reg=LogisticRegression()


# In[58]:


#Training the Support Vector machine Classifier
reg.fit(X_train, Y_train)


# In[59]:


#Model Evaluation: finding accuracy score
#Accuracy score on the traing data


# In[63]:


X_train_predict=reg.predict(X_train)


# In[65]:


training_data_accuracy=accuracy_score(X_train_predict, Y_train)


# In[68]:


print('Accuracy score of the test data :', training_data_accuracy)


# In[71]:


#Accuracy score on the traing data
X_test_predict=reg.predict(X_test)
test_data_accuracy=accuracy_score(X_test_predict, Y_test)


# In[72]:


print('Accuracy score of the test data :', test_data_accuracy)


# In[108]:


# Making a predictive system in which we inut the data and machine will predict, if the patient is diabetic or not
input_data=(4,110,94,1,37.6,0.200,30)

#We will change the input data to the numpy array
input_data_as_numpy_array = np.asarray(input_data)

#we will reshape the array for predicting one instance, so that model will not get confused as total data is not entered
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standarized input data
std_data = SS.transform(input_data_reshaped)

prediction = reg.predict(std_data)
print(prediction)


# In[115]:


if (prediction == 0):
    print('The person is not Diabetic')
else:
    print('The person is Diabetic')


# In[ ]:




