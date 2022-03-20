from sklearn import svm
from matplotlib import pyplot as plt
import pandas as pd
from keras.datasets import mnist
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,classification_report
import datetime as dt
from skimage.transform import resize
from sklearn.model_selection import train_test_split
# data loading
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
#changing size of the images
train_X = resize(train_X,(60000,32,32))
test_X  = resize(test_X,(10000,32,32))

train_X.shape = (60000, 1024)
test_X.shape = (10000, 1024)
target = [0,1,2,3,4,5,6,7,8,9]
# SVM implementation
clf = svm.SVC(C = 1, gamma = 0.001)
clf.fit(train_X,train_Y)
#disp = ConfusionMatrixDisplay.from_estimator(clf,test_X,test_Y,display_labels=target, cmap=plt.cm.Blues,
    #    normalize= None)

#prediction and accuracy
pred = clf.predict(test_X)
accuracy = accuracy_score(test_Y,pred)

disp = ConfusionMatrixDisplay.from_estimator(clf,test_X,test_Y,display_labels=target,cmap=plt.cm.Blues,normalize=None)
plt.title("confusion_matrix_SVM_10000_")
plt.show()
 
