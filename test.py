import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix
from mysklearn import myclassifiers, myevaluation, mypytable, myutils


# recreating the magnet demo we did in class with Sci-kit Learn functions/methods
# so you can see the input and output formats of these functions/methods,
# which will help you implement the functionality to take input of the same form
# and produce output of the same form
# note that the provided unit tests in test_myevaluation.py test against these
# Sci-kit Learn functions/methods so your implementation needs to match Sci-kit
# Learn's format

# little utility function


def display_train_test(label, X_train, X_test, y_train, y_test):
    print("***", label, "***")
    print("train:")
    for i, _ in enumerate(X_train):
        print(X_train[i], "->", y_train[i])
    print("test:")
    for i, _ in enumerate(X_test):
        print(X_test[i], "->", y_test[i])
    print()

# little utility function


def display_folds(label, folds, X, y):
    print(label)
    for i, _ in enumerate(folds):
        curr_fold = folds[i]
        print("fold #:", i)
        train_indexes = list(curr_fold[0])
        test_indexes = list(curr_fold[1])
        train_instances = [str(X[index]) + " -> " + str(y[index])
                           for index in train_indexes]
        test_instances = [str(X[index]) + " -> " + str(y[index])
                          for index in test_indexes]
        print("train indexes:", train_indexes)
        for instance_str in train_instances:
            print("\t" + instance_str)
        print("test indexes:", test_indexes)
        for instance_str in test_instances:
            print("\t" + instance_str)
        print()


# we aren't actually going to use this data for classification,
# just for tracing algorithms that split a dataset into training
# and testing, so I'll make dummy X data where the values of each
# instance make it really clear what its original row index in X
# was (so we can uniquely identify it and watch it move)
# and what it's associated y label is (green or yellow) (so we can
# make sure our Xs and ys stay parallel throughout)
X = [[0, "g"], [1, "g"], [2, "g"], [3, "g"], [4, "g"],
     [5, "y"], [6, "y"], [7, "y"], [8, "y"], [9, "y"]]
# green (pos)/yellow (neg) y labels
y = ["🟢", "🟢", "🟢", "🟢", "🟢", "🟡", "🟡", "🟡", "🟡", "🟡"]

# K FOLD CROSS VALIDATION w/various parameters
# returns lists of indexes
n_splits = 5
standard_kf = KFold(n_splits=n_splits)
folds = list(standard_kf.split(X, y))
X_test1 = []
y_test1 = []
X_train1 = []
y_train1 = []
for i, _ in enumerate(folds):
    curr_fold = folds[i]
    train_indexes = list(curr_fold[0])
    test_indexes = list(curr_fold[1])
    for index in train_indexes:
        X_train1.append(X[index])
        y_train1.append(y[index])
    for index in test_indexes:
        X_test1.append(X[index])
        y_test1.append(y[index])
display_train_test("standard KF CV", X_train1, X_test1, y_train1, y_test1)


X_train_folds_indexes, X_test_folds_indexes = myevaluation.kfold_cross_validation(
    X, 5)

X_test_folds, X_train_folds, y_test_folds, y_train_folds = myutils.indexes_to_fold(
    X_test_folds_indexes, X_train_folds_indexes, X, y)

X_test, X_train, y_test, y_train = myutils.folds_to_train_test(
    X_test_folds, X_train_folds, y_test_folds, y_train_folds)

display_train_test("standard KF CV", X_train, X_test, y_train, y_test)


for val in X_test1:
    if val not in X_test:
        print("X_test1 missing:", val)
for val in X_train1:
    if val not in X_train:
        print("X_train1 missing:", val)
for val in y_test1:
    if val not in y_test:
        print("y_test1 missing:", val)
for val in y_train1:
    if val not in y_train:
        print("y_train1 missing:", val)


print(len(X_test1), len(X_test))
print(len(X_train1), len(X_train))
print(len(y_test1), len(y_test))
print(len(y_train1), len(y_train))
