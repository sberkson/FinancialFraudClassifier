import numpy as np
from mysklearn import myevaluation, myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = (self.regressor.predict(X_test))
        y_predicted = (self.discretizer(y_predicted))

        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # indexes = []
        indexes = myutils.get_row_indexes_distances_sorted(
            self.X_train, X_test)
        distances = []
        neighbor_indexes = []
        for i in range(self.n_neighbors):
            distances.append(indexes[i][1])
            neighbor_indexes.append(indexes[i][0])
        return distances, neighbor_indexes

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        knn_neighbors = self.kneighbors(X_test)
        y_predicted = []
        y_predicted.append(self.y_train[knn_neighbors[1][0]])

        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = myutils.get_most_common_label(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for _ in X_test:
            predictions.append(self.most_common_label)
        return predictions


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = myutils.get_priors(y_train)
        self.posteriors = myutils.get_posteriors(X_train, y_train, self.priors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for row in X_test:
            predictions.append(myutils.get_prediction_naive_bayes(
                row, self.posteriors, self.priors))
        return predictions


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # TODO: programmatically create a header
        #   (e.g. ["att0","att1",...] and create an attribute domains dictionary)
        header = []
        for i in range(len(X_train[0])):
            header.append("att" + str(i))

        # next, i advise stirching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # we are gonna do this because we are going to recursivly traverse the base set, and in CASE 1, we look at the class labels
        # now, we can make a copy of the header, because the tdit algo is going to modify the list
        #   when we split on an attribute, we remove it from the available attributes, because ypu cant split on the same attribute twice
        available_attributes = header.copy()
        # recall: python is pass by object reference
        tree = myutils.tdidt(train, available_attributes, header)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for row in X_test:
            predictions.append(myutils.tdidt_predict(self.tree, row))
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        print("For the class name: " + class_name, " the decision rules are:")
        rules = (myutils.get_all_line_rules(
            self.tree, attribute_names, class_name))
        for rule in rules:
            print(rule)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass  # TODO: (BONUS) fix this


class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        forest(list of MyDecisionTreeClassifier): The list of decision tree classifiers.
        N (int): The number of weak learners.
        M (int): The number of better learners.
        F (int): The number of random attribute subsets.


    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, random_state=None):
        """Initializer for MyRandomForestClassifier.
        """
        self.X = None
        self.y = None
        self.forest = []
        self.N = None
        self.M = None
        self.F = None
        self.random_state = random_state

    def fit(self, X, y, N, M, F):
        """Fits a random forest classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X (list of list of obj): The list of training instances (samples).
                The shape of X is (n_train_samples, n_features)
            y (list of obj): The target y values (parallel to X).
                The shape of y is n_train_samples
            N (int): The number of weak learners.
            M (int): The number of better learners.
            F (int): The number of random attribute subsets.

        """
        self.X = X
        self.y = y
        self.N = N
        self.M = M
        self.F = F

        """
        1. Generate a random stratified test set consisting of one third of the
            original data set, with the remaining two thirds of the instances forming the "remainder set".
        """
        X_test, y_test, X_remainder, y_remainder = myutils.get_test_remainder(
            X, y, random_state=self.random_state)

        # print("X_test", X_test[0])
        # print("y_test", y_test)

        # print("The length of X is: " + str(len(X)))
        # print("The length of X_test is: " + str(len(X_test)))
        # print("The length of X_remainder is: " + str(len(X_remainder)))
        # print("The length of y is: " + str(len(y)))
        # print("The length of y_test is: " + str(len(y_test)))
        # print("The length of y_remainder is: " + str(len(y_remainder)))

        """
        2. Generate N "random" decision trees using bootstrapping
            (giving a training and validation set) over the remainder set.
            At each node, build your decision trees by randomly selecting F
            of the remaining attributes as candidates to partition on.
            This is the standard random forest approach discussed in class.
            Note that to build your decision trees you should still use entropy;
            however, you are selecting from only a (randomly chosen) subset of the available attributes.
        """
        weak_forest = myutils.generate_weak_forest(
            X_remainder, y_remainder, N,  F, random_state=self.random_state)
        # print(weak_forest[0].tree)
        tree_results = {}
        tree_averages = []
        for i, tree in enumerate(weak_forest):
            tree_predicted = tree.predict(X_test)
            tree_results[i] = []
            tree_results[i].append(
                myevaluation.accuracy_score(y_test, tree_predicted))
            tree_results[i].append(
                myevaluation.binary_precision_score(y_test, tree_predicted))
            tree_results[i].append(
                myevaluation.binary_recall_score(y_test, tree_predicted))
            tree_results[i].append(
                myevaluation.binary_f1_score(y_test, tree_predicted))
            tree_averages.append([i, sum(tree_results[i]) / 4])
        # now that we have all ofthe results, we can find the best M ones
        sorted_results = tree_averages.copy()
        sorted_results.sort(key=lambda x: x[1], reverse=True)
        best_trees = sorted_results[:M]
        # print("The best trees are: " + str(best_trees))
        # we know the first index of the best trees is the index of the tree
        strong_forest = []
        for i in best_trees:
            strong_forest.append(weak_forest[i[0]])
        self.forest = strong_forest

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

            Args:
                X_test(list of list of obj): The list of testing samples
                    The shape of X_test is (n_test_samples, n_features)

            Returns:
                y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for row in X_test:
            y_predicted.append(myutils.forest_predict_row(self.forest, row))
        return y_predicted
