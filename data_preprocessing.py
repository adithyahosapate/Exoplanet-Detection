import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.ndimage

#####Imports 

df=pd.read_csv('data/ExoTrain.csv')
df=df.values

data=df[: ,1 :]
labels=df[:,0]

#Train only on positive examples
x_1=data[labels == 1]
x_2=data[labels == 2]

#split into train and test examples
x_train,x_test,y_train,y_test=train_test_split(x_1,np.ones(x_1.shape[0]) ,test_size=0.1)


#normalizing

mean=np.mean(x_train,axis=1)
std= np.std(x_train,axis=1)


x_train=x_train-np.dot(np.array([mean]).T,np.ones([1,x_train.shape[1]]))
x_train=x_train/(np.dot(np.array([std]).T,np.ones([1,x_train.shape[1]])))

test_mean=np.mean(x_test,axis=1)
test_std= np.std(x_test,axis=1)

x_test=x_test-np.dot(np.array([test_mean]).T,np.ones([1,x_test.shape[1]]))
x_test=x_test/(np.dot(np.array([test_std]).T,np.ones([1,x_test.shape[1]])))


#Gaussian filter
x_test=scipy.ndimage.filters.gaussian_filter(x_test,13)
x_train=scipy.ndimage.filters.gaussian_filter(x_train,13)


#normalizing across data points 

mean_data = np.mean(x_train,axis=0)
std_data = np.std(x_train,axis=0)


x_train = (x_train - mean_data)/std_data

test_mean_data = np.mean(x_train,axis=0)
test_std_data = np.std(x_train,axis=0)


x_test = (x_test - test_mean_data)/test_std_data

pca=PCA(n_components=800)
pca.fit(x_train)
x_train=pca.transform(x_train)
x_test=pca.transform(x_test)

import pickle
f = open('train.pickle', 'wb')
pickle.dump((x_train,y_train), f, pickle.HIGHEST_PROTOCOL)
f.close()
f = open('test.pickle', 'wb')
pickle.dump((x_test,y_test), f, pickle.HIGHEST_PROTOCOL)