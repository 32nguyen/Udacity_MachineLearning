import sys
sys.path.append("../Naive Bayes/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here

clf = SVC(kernel="rbf", gamma=500, C=1000)
clf.fit(features_train, labels_train)

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())