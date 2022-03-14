from sklearn import svm
from matplotlib import pyplot as plt
import pandas as pd
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import datetime as dt
from skimage.transform import resize
# data loading
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
#changing size of the images
train_X = resize(train_X,(60000,32,32))
test_X  = resize(test_X,(10000,32,32))

train_X.shape = (60000, 1024)
test_X.shape = (10000, 1024)

# SVM implementation
clf = svm.SVC(C = 1, gamma = 0.001)
start_time = dt.datetime.now()
clf.fit(train_X,train_Y)
end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print('Elapsed time = {}'.format(str(elapsed_time)))
print('/n')
#prediction and accuracy
pred = clf.predict(test_X)
accuracy = accuracy_score(test_Y,pred)
print(accuracy)
 
