#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:57:16 2018

@author: liangzhouji
"""

import pandas as pd
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score

###Import Chandon data 
#B2M
d = scipy.io.loadmat("B2M_Dip.mat")
s = scipy.io.loadmat("B2M_SOS.mat")
t = scipy.io.loadmat("B2M_Tensor.mat")



dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

B2M =  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#M2I
d = scipy.io.loadmat("M2I_Y_Dip.mat")
s = scipy.io.loadmat("M2I_Y_SOS.mat")
t = scipy.io.loadmat("M2I_Y_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

M2I =  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#I2D
d = scipy.io.loadmat("I2D_Dip.mat")
s = scipy.io.loadmat("I2D_SOS.mat")
t = scipy.io.loadmat("I2D_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

I2D=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#L2B
d = scipy.io.loadmat("L2B_Dip.mat")
s = scipy.io.loadmat("L2B_SOS.mat")
t = scipy.io.loadmat("L2B_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

L2B=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#L2M
d = scipy.io.loadmat("L2M_Dip.mat")
s = scipy.io.loadmat("L2M_SOS.mat")
t = scipy.io.loadmat("L2M_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

L2M=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#M2B
d = scipy.io.loadmat("M2B_Dip.mat")
s = scipy.io.loadmat("M2B_SOS.mat")
t = scipy.io.loadmat("M2B_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

M2B=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)


#T2U
d = scipy.io.loadmat("T2U_Dip.mat")
s = scipy.io.loadmat("T2U_SOS.mat")
t = scipy.io.loadmat("T2U_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

T2U=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#M2I
d = scipy.io.loadmat("M2I_Dip.mat")
s = scipy.io.loadmat("M2I_SOS.mat")
t = scipy.io.loadmat("M2I_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

M2I=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#Y_M2I
d = scipy.io.loadmat("M2I_Y_Dip.mat")
s = scipy.io.loadmat("M2I_Y_SOS.mat")
t = scipy.io.loadmat("M2I_Y_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

Y_M2I=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#Y_M2B
d = scipy.io.loadmat("M2B_Y_Dip.mat")
s = scipy.io.loadmat("M2B_Y_SOS.mat")
t = scipy.io.loadmat("M2B_Y_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

Y_M2B=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)

#Y_L2M
d = scipy.io.loadmat("I2M_Y_Dip.mat")
s = scipy.io.loadmat("I2M_Y_SOS.mat")
t = scipy.io.loadmat("I2M_Y_Tensor.mat")

dip = d['data']
sos = s['data']
tensor = t['data']

dip = dip.reshape(dip.size,1)
sos = sos.reshape(sos.size,1)
tensor = tensor.reshape(tensor.size,1)

Y_L2M=  np.concatenate((sos,np.concatenate((tensor,dip),axis=1)),axis=1)


# Create an empty dataframe
lst = pd.DataFrame({'SOS':[],'Tensor':[],'Dip':[],'label':[]})

#dataList=[A2B, B2T, B2M, C2L, D2A, I2D, L2B, L2M, M2I, M2B, T2U, U2L]
#datastr=["A2B", "B2T", "B2M", "C2L", "D2A", "I2D", "L2B", "L2M", "M2I", "M2B", "T2U", "U2L"]
# import the data into dataframe
dataList=[B2M, M2I, M2B, I2D, L2B, L2M ,T2U]
datastr=["B2M", "M2I", "M2B",  "I2D", "L2B", "L2M" ,"T2U"]

for i in range(len(datastr)):
    features = dataList[i]# extract the data
    att = pd.DataFrame(features)# create an empty dataframe to store each attributes
    att = att[(att.T != 0).any()] # delete rows with only 0 
    att.columns = ['SOS','Tensor','Dip'] # name each column
    att['label'] = datastr[i] # add label
    lst = lst.append(att) # merge all data

#reset index
lst = lst.reset_index(drop = True)

#Encode label
le = LabelEncoder().fit(lst["label"])
y_train = le.transform(lst["label"])
X_train = lst.drop("label", axis=1)
print(le.classes_)

#SMO = SMOTE()
#X_resampled, y_resampled = SMO.fit_sample(X_train , y_train)
#label1 = le.inverse_transform(y_resampled)

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X_train , y_train)
label1 = le.inverse_transform(y_resampled)

X_scaled = preprocessing.scale(X_resampled)
scaler = preprocessing.StandardScaler().fit(X_resampled)

#
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, train_size = 0.8, test_size=0.2, random_state=0)

##### Step one Validation: Vlassification accuracy on each data point
#for i in range(200):  
#    knn = KNeighborsClassifier(n_neighbors= (i+9))
#    knn.fit(X_train, y_train)
#    
#    prediction_train = knn.predict(X=X_train)
#    prediction_test = knn.predict(X=X_test)
#    print('k=', i+9, 'Classification accuracy on training set: {:.3f}'.format(accuracy_score(y_train,prediction_train)))
#    print('k=', i+9, 'Classification accuracy on test set: {:.3f}'.format(accuracy_score(y_test,prediction_test)))

##
def counter(arr):   # define function to account number of data in each label
    return Counter(arr).most_common(7)

knn = KNeighborsClassifier(n_neighbors= 100)
knn.fit(X_scaled, y_resampled)

###### under sample step two validation
#for i in range(len(datastr)):
#    a = datastr[i]
#    df = lst[lst['label'].isin([a])]
#    df.drop(["label"], axis = 1, inplace = True)
#    df=df.sample(n= X_scaled.shape[0], replace = True, random_state = 1)
#    
#    df = scaler.transform(df)    
#    
#    test = knn.predict(df)
#      
#    count = []
#        
#    x = counter(test)
#    
#    for j in range(len(counter(test))):
#        l = list(x[j])
#        l[0] = le.classes_[l[0]]
#        count.append(l)
#    
#    print("Prediction on test set",a,"is",count)


#####Over sample step two validation
#label1 = le.inverse_transform(y_test)
#X_test = pd.DataFrame(X_test)
#X_test.columns = ['SOS','Tensor','Dip']
#X_test["label"] = label1

#for i in range(len(datastr)):
#    a = datastr[i]
#    df = X_test[X_test['label'].isin([a])]
#    df.drop(["label"], axis = 1, inplace = True)
#    
#    test = knn.predict(df)
#      
#    count = []
#       
#    x = counter(test)
#    
#    for j in range(len(counter(test))):
#        l = list(x[j])
#        l[0] = le.classes_[l[0]]
#        count.append(l)
#    
#    print("Prediction on test set",a,"is",count)

#####Test code

dataList1=[Y_L2M, Y_M2B, Y_M2I]
datastr1=["Y_L2M", "Y_M2B", "Y_M2I"]

Testlst = pd.DataFrame({'SOS':[],'Tensor':[],'Dip':[],'label':[]})

for i in range(len(datastr1)):
    features = dataList1[i]
    att = pd.DataFrame(features)
    att = att[(att.T != 0).any()] # delete rows with only 0 
    att.columns = ['SOS','Tensor','Dip'] # name each column
    att['label'] = datastr1[i] # add label
    Testlst = Testlst.append(att) # merge all data


for i in range(len(datastr1)):
    a = datastr1[i]
    df = Testlst[Testlst['label'].isin([a])]
    df.drop(["label"], axis = 1, inplace = True)
    df=df.sample(n= X_scaled.shape[0], replace = True, random_state = 1)
    
    df = scaler.transform(df)    
    
    test = knn.predict(df)
      
    count = []
        
    x = counter(test)
    
    ##inverse label counter
    for j in range(len(counter(test))):
        l = list(x[j]) # convert tuple to list
        l[0] = le.classes_[l[0]]
        count.append(l)
    
    print("Prediction on test set",a,"is",count)
    
    

