from cgi import test
import csv
import pandas as pd
from skimage.transform import resize

# MNIST
# train = pd.read_csv('mnist\mnist_train.csv')
# test = pd.read_csv('mnist\mnist_test.csv')

# train_X = train.drop('label', axis=1)
# train_Y = train['label']
# test_X = test.drop('label', axis=1)
# test_Y = test['label']

# train_X_reshape = train_X.values.reshape(-1,28,28)
# train_X_resize = resize(train_X_reshape,(60000,32,32))

# test_X_reshape = test_X.values.reshape(-1,28,28)
# test_X_resize = resize(test_X_reshape,(10000,32,32))


# fashion-mnist
fashion_train = pd.read_csv('mnist-fashion\mshion-mnist_train.csv')
fashion_test = pd.read_csv('mnist-fashion\mshion-mnist_test.csv')

Ftrain_X = fashion_train.drop('label', axis=1)
Ftrain_Y = fashion_train['label']
Ftest_X = fashion_test.drop('label', axis=1)
Ftest_Y = fashion_test['label']

Ftrain_X_reshape = Ftrain_X.values.reshape(-1, 28, 28)
Ftrain_X_resize = resize(Ftrain_X_reshape, (60000, 32, 32))

Ftest_X_reshape = Ftest_X.values.reshape(-1, 28, 28)
Ftest_X_resize = resize(Ftest_X_reshape, (10000, 32, 32))
