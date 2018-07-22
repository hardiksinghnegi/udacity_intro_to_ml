#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]



#########################################################
### your code goes here ###

email_svm = SVC(C=10000.0,kernel="rbf")
email_svm.fit(features_train,labels_train)

author_pred = email_svm.predict(features_test)

# print author_pred[10]
# print author_pred[26]
# print author_pred[50]

chris_counter = 0
for x in author_pred:
    if x == 1:
        chris_counter +=1

print chris_counter

# print accuracy_score(labels_test,author_pred)

#########################################################


