#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
 ### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
clf.fit(features_train, labels_train)

### use the trained classifier to predict labels for the test features
pred = clf.predict(features_test)

### Print the accuracy
### accuracy = no. of points classified correctly / all points (in test set)
print accuracy_score(pred, labels_test)
#########################################################


