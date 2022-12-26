#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras 
from sklearn.metrics import accuracy_score, classification_report 
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['target'] = pd.Series(iris.target)
df.head(5)


# In[3]:


df.shape


# In[4]:


plt.scatter(x = df['sepal length (cm)'], y = df['sepal width (cm)'], c =  df['target'] )
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title('sepal width vs length')


# In[5]:


plt.scatter(x = df['petal length (cm)'], y = df['petal width (cm)'], c = df['target'])
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('petal width vs length')


# In[6]:


X = df.drop(columns = ['target'])
y = tf.keras.utils.to_categorical(df['target'],3) 
# Note : if you schoose not to do categoreical encoding change loss armunel in compile to sparse_categorical_crossentropy
#y = df['target']
print('X shape', X.shape)
print('y shape', y.shape)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)
print('X_train shape:', X_train.shape, '   X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape, '   y_test shape:', y_test.shape)


# In[8]:


#model 

model = keras.Sequential([
     keras.layers.Dense(25, input_shape = (4,), activation = 'relu'),
     keras.layers.Dense(15,activation = 'relu'),
     keras.layers.Dense(3, activation = 'softmax')
])

model.compile( optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 50)


# In[9]:


model.summary()


# In[10]:


y_pred = pd.DataFrame(model.predict(X_test)) # we get probabilties for each class
y_pred = y_pred.idxmax(axis = 1) # taking the class with max probabilty
y_pred[:5]


# In[11]:


model.evaluate(X_test,y_test)


# In[13]:


y_test = pd.DataFrame(y_test).idxmax(axis = 1) # earlier did one hot encoding on y thus reversing tha


# In[14]:


accuracy_score(y_test, y_pred)


# In[15]:


cm = tf.math.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)


# In[17]:


clf = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(clf), annot = True)


# In[ ]:




