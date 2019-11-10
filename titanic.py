# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:56:31 2019

@author: olegm
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
submission = pd.read_csv('gender_submission.csv')


    
#extracting some data
x_train = train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
#dependant variable
y_train = train.loc [:, "Survived"]


#Filling NaN values with zeroes

train_age = x_train.loc[:, "Age"]
x_train.loc[:, "Age"] = train_age.fillna(train_age.mean())

encoder = LabelEncoder()
"""
Encoding gender to binary values -> male = 1, female = 0
"""
x_train.loc[:,"Sex"] = encoder.fit_transform(x_train.loc[:,"Sex"])


#Joining two dataframes together by passanger ID
test_formed = pd.merge(left= test, right=submission, how="left", left_on="PassengerId", right_on="PassengerId")
x_test = test_formed.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

#Filling NaN values with zeroes
test_age = x_test.loc[:, "Age"]
x_test.loc[:, "Age"] = test_age.fillna(test_age.mean())

x_test.loc[:, "Fare"] = x_test.loc[:, "Fare"].fillna(x_test.loc[:, "Fare"].mean())

encoder2 = LabelEncoder()
"""
Encoding gender to binary values -> male = 1, female = 0
"""
x_test.loc[:,"Sex"] = encoder2.fit_transform(x_test.loc[:,"Sex"])

y_test = test_formed.loc[:, "Survived"]






"""
Machine learning part
"""


"""
Clustering
"""
#cluster CLassifier
from sklearn.cluster import KMeans

#fitting model
model = KMeans(n_clusters = 2, max_iter=1000)
model.fit(x_train,y_train)

prediction = model.predict(x_test)

from sklearn.metrics import confusion_matrix 
matrix = confusion_matrix(y_test,prediction)

#result [[254,12],
#        [133,19]] - not accurate


"""
Linear regression
"""

from sklearn.linear_model import LinearRegression

modelLinReg = LinearRegression()
modelLinReg.fit(x_train,y_train)

linearPredict = modelLinReg.predict(x_test)
#linear regression in this case will return the euclidian distance from the 0 point
#our calculated point is between 0 and 1

# 0 I-------------X---I  1    --> simple graph, if this returns propabilty,
# guess which one is more propable   

for i in range(0, linearPredict.size):
    linearPredict[i] = 1 if linearPredict[i] >= 0.5 else 0
    
linearRegMatrix= confusion_matrix(y_test,linearPredict)
#result [[261,5],
#        [6,146]] -- really accurate


"""
Decision Tree
"""

from sklearn.tree import DecisionTreeClassifier

modelTreeClass = DecisionTreeClassifier();
modelTreeClass.fit(x_train,y_train)

treeClassPrediction = modelTreeClass.predict(x_test)

treeClassMatrix = confusion_matrix(y_test,treeClassPrediction)
#result [[225,41]
#        [44,108]] -- mediocre result


"""
Random Forest
"""
from sklearn.ensemble import RandomForestClassifier

modelForestClass = RandomForestClassifier()
modelForestClass.fit(x_train,y_train)

modelForestPrediction = modelForestClass.predict(x_test)

forestClassMatrix = confusion_matrix(y_test,modelForestPrediction)
#result [[236,30],
#        [49,103]] -- mediocre result

"""
Naive Bayes
"""

from sklearn.naive_bayes import GaussianNB

modelNaiveBay = GaussianNB()
modelNaiveBay.fit(x_train,y_train)
naiveBayPred = modelNaiveBay.predict(X=x_test)

bayesMatrix = confusion_matrix(y_test,naiveBayPred)
# result [[243,23],
#         [6,146]]


#I'm lazy lets automate this task

def accuracyOnModel(x_train,y_train,x_test,y_test, model, model_set = False):
    if not model_set:
        construct = model()
    else:
        construct = model
    construct.fit(x_train,y_train)
    prediction = construct.predict(X=x_test)
    
    confMatrix = confusion_matrix(y_test,prediction)
    return confMatrix,prediction

from sklearn.svm import LinearSVC
linearSVCMatrix, LinearSVCPred = accuracyOnModel(x_train,y_train,x_test,y_test,LinearSVC(max_iter=3000),True)
# result[[265,1],
#        [11,141]] -- accurate



