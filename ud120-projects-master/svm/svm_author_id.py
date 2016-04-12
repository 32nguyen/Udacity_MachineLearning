#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.append("../../Lecture 1/")
sys.path.append("../tools")

from time import time

from email_preprocess import preprocess
from sklearn.svm import SVC


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
#########################################################

clf = SVC(kernel="rbf", C=10000)
clf.fit(features_train, labels_train)
pre = clf.predict(features_test)

count = 0.0
for i in range(0, len(pre)):
    if pre[i] == 1:
        count += 1
print count
accuracy = clf.score(features_test, labels_test)
print 'the accruracy is :', accuracy

#########################################################


