from sklearn import svm
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
import numpy as np
from reader import Ftrain_X_resize as train_X,Ftrain_Y as train_Y, Ftest_X_resize as test_X,Ftest_Y as test_Y
# from reader import train_X_resize as train_X,train_Y, test_X_resize as test_X,test_Y
def mnist_knn_full_fromat(train_X,train_Y,test_X,test_Y):
    """
    function does KNN clasification with 60000 train images and 10000 test images
    """
    #changing size of the images
    # train_X = resize(train_X,(60000,32,32))
    # test_X  = resize(test_X,(10000,32,32))
    target = [0,1,2,3,4,5,6,7,8,9]
    train_X.shape = (60000, 1024)
    test_X.shape = (10000, 1024)

    # KNN implementation
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(train_X,train_Y)

    #prediction and accuracy
    pred = clf.predict(test_X)
    accuracy = accuracy_score(test_Y,pred)*100
    print(classification_report(test_Y,pred))
    disp = ConfusionMatrixDisplay.from_estimator(clf,test_X,test_Y,display_labels=target,cmap=plt.cm.Blues,normalize=None)
    plt.title("confusion_matrix_KNN_10000_samples")
    plt.show()

def  mnist_knn_small_fromat(training_images):
    """
    function does KNN clasification with deacresed number of images 
    """
    #data loading and processing
    train = pd.read_csv(training_images)
    train = train.groupby('label').head(40)
    target = train['label']
    class_names = [0,1,2,3,4,5,6,7,8,9]
    values = train.values
    data = np.delete(values,0,1)
    
    #resize 28x28 images to 32x32 images
    data = data.reshape(400,28,28)
    # print(data.shape)
    data32x32 = resize(data,(400,32,32))
    datax1024 = data32x32.reshape(400,1024)
    train_X,test_X,train_Y,test_Y = train_test_split(datax1024,target,test_size=0.5,)

    #KNN implementation
    clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(train_X,train_Y)

    #prediction and accuracy
    pred = clf.predict(test_X)
    accuracy = accuracy_score(test_Y,pred)*100
    #print(classification_report(test_Y, pred))

    disp = ConfusionMatrixDisplay.from_estimator(clf,test_X,test_Y,display_labels=class_names,cmap=plt.cm.Blues,normalize=None)
    plt.title("confusion_matrix_KNN_400_samples")
    plt.show()

#mnist_knn_small_fromat("datasets/train.csv")
mnist_knn_full_fromat(train_X,train_Y,test_X,test_Y)