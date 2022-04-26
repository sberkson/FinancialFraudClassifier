from mysklearn import myutils
from random import random
import numpy as np


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if shuffle:
        X, y = myutils.randomize_in_place(X, y, random_state)
    if isinstance(test_size, float):
        n_test = int(test_size * len(X)) + 1
    else:
        n_test = test_size

    X_train = X[:-n_test]  # makes the training set from [0, n_test)
    y_train = y[:-n_test]
    X_test = X[-n_test:]  # makes the testing set from [n_test, len(X))
    y_test = y[-n_test:]

    return X_train, X_test, y_train, y_test


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_temp = X.copy()  # this may or may not be shuffled
    if shuffle:
        X_temp = myutils.randomize_in_place(X_temp, seed=random_state)
    X_train_folds = []
    X_test_folds = []
    # # we know there will be n_splits folds, so we can just make a list of n_splits empty lists
    for i in range(n_splits):
        X_train_folds.append([])
        X_test_folds.append([])
    # for i in range(n_splits):
    for i in range(n_splits):
        for j in range(len(X)):
            if i == j % n_splits:
                X_test_folds[i].append(X_temp.index(X[j]))
            else:
                X_train_folds[i].append(X_temp.index(X[j]))
    return X_train_folds, X_test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_temp = X.copy()  # this may or may not be shuffled
    X_train_folds = []
    X_test_folds = []
    # # we know there will be n_splits folds, so we can just make a list of n_splits empty lists
    for i in range(n_splits):
        X_test_folds.append([])
    # now we make the folds,
    group_val_names, group_val_index = myutils.group_by(y)

    for i in range(n_splits):  # create the folds
        # inside each fold (lenght of X)
        for j in range(len(group_val_index[0])):
            try:
                X_test_folds[j % n_splits].append(group_val_index[i][j])
            except IndexError:
                pass
    # now we can do the training (all the values not in each fold)
    for i in range(n_splits):
        X_train_folds.append([])
        for j in range(len(X)):
            if j not in X_test_folds[i]:
                X_train_folds[i].append(X_temp.index(X[j]))
    if shuffle:
        print("Before:", X_train_folds)
        print("Before:", X_test_folds)
        # this is needed for some reason (the +1)
        myutils.randomize_in_place(X_train_folds, X_test_folds, random_state+1)
        print("After:", X_train_folds)
        print("After:", X_test_folds)
    return X_train_folds, X_test_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if n_samples is None:
        n_samples = len(X)
    if random_state is None:
        random_state = int(random())
    np.random.seed(random_state)
    X_sample = []
    X_out_of_bag = []
    if y is None:
        y_sample = None
        y_out_of_bag = None
    else:
        y_sample = []
        y_out_of_bag = []

    for i in range(n_samples):
        # randomly sample a sample from the dataset
        sample_index = np.random.randint(len(X))
        X_sample.append(X[sample_index])
        if y is not None:
            y_sample.append(y[sample_index])

    # now we can find the leftovers
    for i in range(len(X)):
        if X[i] not in X_sample:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for i in range(len(labels)):
        matrix.append([0] * len(labels))
    for i, y_val in enumerate(y_true):  # pylint wanted an enumerate here
        if y_val in labels:
            matrix[labels.index(y_val)][labels.index(y_pred[i])] += 1
    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_predictions = 0
    # pylint wanted me to use an enumerate here
    for i, prediction in enumerate(y_pred):
        if prediction == y_true[i]:
            correct_predictions += 1
    if not normalize:
        return correct_predictions
    return correct_predictions / len(y_true)


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = np.unique(y_true)
    if pos_label is None:
        pos_label = labels[0]
    true_positives = 0
    false_positives = 0
    for i, prediction in enumerate(y_pred):  # pylint wanted me to use an enumerate here :(
        if prediction == pos_label and y_true[i] == pos_label:
            true_positives += 1
        elif prediction == pos_label and y_true[i] != pos_label:
            false_positives += 1
    if true_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = np.unique(y_true)
    if pos_label is None:
        pos_label = labels[0]
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i, prediction in enumerate(y_pred):  # pylint wanted me to use an enumerate here :(
        if prediction == pos_label and y_true[i] == pos_label:
            true_positives += 1
        elif prediction == pos_label and y_true[i] != pos_label:
            false_positives += 1
        elif prediction != pos_label and y_true[i] == pos_label:
            false_negatives += 1
    if true_positives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = np.unique(y_true)
    if pos_label is None:
        pos_label = labels[0]
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
