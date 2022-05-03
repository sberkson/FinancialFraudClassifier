from mysklearn.mypytable import MyPyTable

from sklearn.linear_model import LinearRegression
from sqlalchemy import false
from mysklearn.myclassifiers import (
    MyDecisionTreeClassifier,
    MyNaiveBayesClassifier,
    MySimpleLinearRegressionClassifier,
    MyKNeighborsClassifier,
    MyDummyClassifier,
    MyRandomForestClassifier,
)


def test_myforest_fit():
    forest = MyRandomForestClassifier(random_state=0)
    titanic = MyPyTable().load_from_file("input_data/titanic.csv")
    y = titanic.get_column("survived")
    titanic.drop_col("survived")
    X = titanic.data

    forest.fit(X, y, 100, 10, 2)

    assert forest.forest[0].tree == ['Attribute', 'att2', ['Value', 'male', [
        'Leaf', 'no', 1, 2]], ['Value', 'female', ['Leaf', 'yes', 1, 2]]]
    assert forest.forest[1].tree == ['Attribute', 'att2', ['Value', 'female', [
        'Leaf', 'yes', 1, 2]], ['Value', 'male', ['Leaf', 'no', 1, 2]]]
    assert forest.forest[2].tree == ['Attribute', 'att2', ['Value', 'female', [
        'Leaf', 'yes', 1, 2]], ['Value', 'male', ['Leaf', 'no', 1, 2]]]
    assert forest.forest[3].tree == ['Attribute', 'att2', ['Value', 'female', [
        'Leaf', 'yes', 1, 2]], ['Value', 'male', ['Leaf', 'no', 1, 2]]]
    assert forest.forest[4].tree == ['Attribute', 'att1', ['Value', 'adult', [
        'Leaf', 'no', 1, 2]], ['Value', 'child', ['Leaf', 'yes', 1, 2]]]
    assert forest.forest[5].tree == ['Attribute', 'att0', ['Value', 'crew', [
        'Leaf', 'no', 1, 2]], ['Value', 'third', ['Leaf', 'yes', 1, 2]]]
    assert forest.forest[6].tree == ['Attribute', 'att0', ['Value', 'third', [
        'Leaf', 'yes', 1, 2]], ['Value', 'crew', ['Leaf', 'no', 1, 2]]]
    assert forest.forest[7].tree == ['Attribute', 'att0', ['Value', 'crew', [
        'Leaf', 'no', 1, 2]], ['Value', 'third', ['Leaf', 'yes', 1, 2]]]
    assert forest.forest[8].tree == ['Attribute', 'att0', ['Value', 'third', [
        'Leaf', 'yes', 1, 2]], ['Value', 'crew', ['Leaf', 'no', 1, 2]]]
    assert forest.forest[9].tree == ['Attribute', 'att0', ['Value', 'third', [
        'Leaf', 'yes', 1, 2]], ['Value', 'crew', ['Leaf', 'no', 1, 2]]]


def test_myforest_predict():
    forest = MyRandomForestClassifier(random_state=0)
    titanic = MyPyTable().load_from_file("input_data/titanic.csv")
    y = titanic.get_column("survived")
    titanic.drop_col("survived")
    X = titanic.data

    forest.fit(X, y, 100, 10, 2)
    X_test = [['crew', 'adult', 'female'], ['first', 'adult', 'male'], [
        'first', 'adult', 'male'], ['third', 'adult', 'female']]
    y_predicted = forest.predict(X_test)
    y_actual = ['no', 'no', 'no', 'yes']
    assert y_predicted == y_actual
