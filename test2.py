import importlib
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix
from mysklearn import myclassifiers, myevaluation, mypytable, myutils


importlib.reload(mypytable)
# first we are going to import the dataset into a mypytable object
mytable = mypytable.MyPyTable()
mytable.load_from_file("Fraud_chop.csv")

print(mytable.column_names)

print(mytable.data[0])

# we know from my datachoping notebook what each column is and does
# mytable.drop_cols(['step','type','nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'])
# we dont need step, nameOrig, nameDest, isFlaggedFraud
mytable.drop_cols(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
# this is because:
# step: this is just the time from the start of the data collection
# nameOrig: this is the name of the person who sent the money
# nameDest: this is the name of the person who received the money
# isFlaggedFraud: this is the classification results for the AI that the group that collected the data created

print(mytable.data[0])
print(mytable.column_names)

# the values of type are strings, so we will convert them to ints to be able to be used in the classifiers
mytable.convert_col_to_int('type')

data = mytable.data
headers = mytable.column_names

# we also will make x and y
X = []
y = []
for row in data:
    X.append(row[0:len(row)-1])
    y.append(row[-1])


forest_clf = myclassifiers.MyRandomForestClassifier()
forest_clf.fit(X, y, 100, 10, 2)
