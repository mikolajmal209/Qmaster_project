from pickletools import dis
from sklearn import svm
from matplotlib import pyplot as plt
import pandas as pd
from keras.datasets import mnist
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
import numpy as np
from sympy import false
# data loading
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
#changing size of the images
train_X = resize(train_X,(60000,32,32))
test_X  = resize(test_X,(10000,32,32))
target = [0,1,2,3,4,5,6,7,8,9]
train_X.shape = (60000, 1024)
test_X.shape = (10000, 1024)


# SVM implementation
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(train_X,train_Y)

#prediction and accuracy
pred = clf.predict(test_X)
accuracy = accuracy_score(test_Y,pred)*100

disp = ConfusionMatrixDisplay.from_estimator(clf,test_X,test_Y,display_labels=target,cmap=plt.cm.Blues,normalize=None)
plt.title("confusion_matrix_KNN_10000_samples")
plt.show()
