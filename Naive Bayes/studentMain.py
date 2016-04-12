#!/usr/bin/python
from ClassifyNB import classify
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy
from sklearn.metrics import accuracy_score
features_train, labels_train, features_test, labels_test = makeTerrainData()

# the training data (features_train, labels_train) have both "fast" and "slow" points mixed
# in together--separate them so we can give them different colors in the scatterplot,
# and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

clf = classify(features_train, labels_train)

# draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

# compare tha accuracy by hand
count = 0.0  # have to be in floating point
predict = clf.predict(features_test)
for i in range(0, len(predict)):
    if predict[i] == labels_test[i]:
        count += 1
accuracy = count/(len(predict))
print 'self-calculate: ', accuracy
# using built-in function of sklearn
acc = NBAccuracy(features_train, labels_train, features_test, labels_test)
print 'using clf.score: ', acc
# or
print 'accuracy_score function', accuracy_score(predict,labels_test)




