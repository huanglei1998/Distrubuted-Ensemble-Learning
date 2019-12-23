from sklearn import svm

import pandas as pd
import numpy as np
import random

dataframe = pd.read_csv("phoneme.dat",header=None,skiprows=[0,1],sep='\s+')
data=dataframe.values
random.shuffle(data)

for i in range(data.shape[0]):
    if(data[i,-1]==0):
        data[i,-1]=-1

train_data=data[0:5000,:]
test_data=data[5000:,:]
train_x=train_data[:,0:-1]
train_y=train_data[:,-1]
test_x=test_data[:,0:-1]
test_y=test_data[:,-1]

w = np.ones(train_y.shape)
weight = w / train_y.shape[0]

svm_cla = svm.SVC(kernel="sigmoid",probability=True)
svm_cla.fit(train_x,train_y,sample_weight=weight)

predict_y=svm_cla.predict_proba(test_x)
score=svm_cla.score(test_x,test_y)
print(score)




