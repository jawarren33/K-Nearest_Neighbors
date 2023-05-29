#!/usr/bin/env python
# coding: utf-8

# We are going to be working with the **Wine** dataset. 
# This is a 178 sample dataset that categorises 3 different types of Italian wine using 13 different features. 
# The code below loads the Wine dataset and selects a subset of features to work with. 


# set matplotlib backend to inline
get_ipython().run_line_magic('matplotlib', 'inline')

# import modules
from sklearn import datasets 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

# load data
wine=datasets.load_wine()
#print(wine.DESCR)
# this dataset has 13 features, we will only choose a subset of these
df_wine = pd.DataFrame(wine.data, columns = wine.feature_names )
selected_features = ['alcohol','flavanoids','color_intensity','ash']

# extract the data as numpy arrays of features, X, and target, y
y = wine.target
X = pd.DataFrame((df_wine[selected_features].values), columns = (selected_features))
X_val = np.array(X)
print(X)


# The first part of tackling any ML problem is visualising the data in order to understand some of the properties of the problem at hand. 
# When there are only a small number of classes and features, it is possible to use scatter plots to visualise interactions between different pairings of features. 
# 
# 
# A function that, given data X and labels y, plots this grid.  The function can be invoked like this:
#         
#     myplotGrid(X,y,...)
#     
# where X is the training data and y are the labels (you may also supply additional optional arguments). 


# define plotting function 
def myplotGrid(X,y):
    X['target']=y
    
    sns.pairplot(X, hue = "target" , markers=['o','s','d'])
    del X['target']
    return          
        

# run the plotting function
myplotGrid(X,y)


# 
# Add some Gaussian Noise:

mySeed = 12345
np.random.seed(mySeed)
XNoise = np.random.normal(0,0.5,X.shape)
XN = pd.DataFrame(X + XNoise)
XN_Val = np.array(XN)

print(XN)
print(XN.shape)

#graph noise data
myplotGrid(XN,y)


# 
# Function that performs k-NN given a set of data and can be invoked by:
# 
#         y_ = mykNN(X,y,X_,options)
#         
# where X is the training data, y is the training outputs, X\_ are the testing data and y\_ are the predicted outputs for X\_.  The options argument (can be a list or a set of separate arguments depending on how you choose to implement the function) should at least contain the number of neighbours to consider as well as the distance function employed.
# 
# 


# helper code for X
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_val, y, test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(y_test)
print(y_pred) 


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print("For clean data: \n", confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



#helper code for XN
XN_train, XN_test, yN_train, yN_test = train_test_split(XN_Val, y, test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(XN_train,yN_train)
yN_pred=knn.predict(XN_test)
print(yN_test)
print(yN_pred) 

print("For noisy data: \n", confusion_matrix(yN_test,yN_pred))
print(accuracy_score(yN_test,yN_pred))



#Split the data into a train and test set for clean data

import random

#Clean the data
def CleanX(X):
    data=X.values
    return data[:,1:]

#Split Data 
def Split(X,Y):
    print(Y)
    shuffler = np.random.permutation(X.shape[0])
    X_shuffled = X[shuffler]
    Y_shuffled = Y[shuffler]
    print(Y_shuffled)
    split = int(0.8*X.shape[0])

    X_train =X_shuffled[:split,:]
    Y_train = Y_shuffled[:split]

    X_test = X_shuffled[split:,:]
    Y_test = Y_shuffled[split:]
    return X_train,Y_train,X_test,Y_test



#define a distance function between the training and test points
#define 2 distance functions to test in cross validation
def euclidian_dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def manhattan_dist(x1,x2):
    return np.abs(sum(x1-x2))


#define neighbors function 

def kNN(X,Y,queryPoint,k,dist):
    #create empty list to store distance values
    vals = []
    m = X.shape[0]
    d=0
    
    #iterate through to find which distance to use
    for i in range(m):
        if dist == 0:
            d = euclidian_dist(queryPoint,X[i])
        elif dist==1:
            d = manhattan_dist(queryPoint,X[i])
        
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    
    # Nearest/First K points
    
    vals = vals[:k]
    
    vals = np.array(vals)
    
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


#assign prediction labels to test sample

def knn_pred(X,Y,X_,k,d):
    y_pred=[]
    for x in X_:
        y_pred.append(kNN(X,Y,x,k,d))
    return y_pred



def myknn(X_df,Y,k,d):
    X = CleanX(X_df)
    X,Y,X_,Y_=Split(X,Y)
    return knn_pred(X,Y,X_,k,d),Y_


ypred,ytest=myknn(X,y,5,0)
ypredN,ytestN=myknn(XN,y,5,0)
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytestN,ypredN))

     

# 
# In the cell below, I implement my own classifier evaluation code. 

#Accuracy Score
def accuracy(ytest,ypred):
    accscore = (sum(abs(ytest-ypred))/len(ypred))
    
    return 1 - accscore

print("The Accuracy score for the clean data is: \n", accuracy(ytest,ypred)*100, "%")
print("The Accuracy score for the Noisy data is: \n", accuracy(ytestN,ypredN)*100, "%")


#Confusion Matrix
def myconfmat(ytest,ypred):
    cm = np.zeros((3,3), dtype=np.int)
    for i in range(0,len(ypred)):
        cm[ytest[i],ypred[i]]+=1
    return cm

print("The confusion matrix for the clean data is : \n" , myconfmat(y_test,y_pred) )


