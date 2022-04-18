import math
import operator
import numpy as np


def compute_euclidean_distance(v1, v2):
    """Computes the Euclidean distance between two vectors.
    Args:
        v1(list of numeric vals): The first vector
        v2(list of numeric vals): The second vector
    Returns:
        distance(float): The Euclidean distance between v1 and v2
    """
    total = 0
    for i in range(len(v1)):
        total += (v1[i] - v2[i]) ** 2
    return total ** (1/2)


def get_row_indexes_distances_sorted(X_train, X_test):
    """Computes the distances between each test instance in X_test and each
        training instance in X_train, and returns the sorted list of
        (index, distance) tuples.
    Args:
        X_train(list of list of numeric vals): The list of training instances (samples).
            The shape of X_train is (n_train_samples, n_features)
        X_test(list of list of numeric vals): The list of testing samples
            The shape of X_test is (n_test_samples, n_features)
    Returns:
        sorted_distances(list of (int, float)): The list of (index, distance) tuples
            sorted in ascending order by distance
    """
    row_indexes_distances = []
    for i, train_instance in enumerate(X_train):
        dist = compute_euclidean_distance(train_instance, X_test)
        row_indexes_distances.append([i, dist])
    # now we can sort the items by the distance (item[1])
    row_indexes_distances.sort(key=operator.itemgetter(-1))
    return row_indexes_distances


def get_mpg_rating(mpg):
    """
    Return the rating of the given mpg
    """
    if mpg >= 45:
        rating = 10
    elif 37 <= mpg < 45:
        rating = 9
    elif 31 <= mpg < 37:
        rating = 8
    elif 27 <= mpg < 31:
        rating = 7
    elif 24 <= mpg < 27:
        rating = 6
    elif 20 <= mpg < 24:
        rating = 5
    elif 17 <= mpg < 20:
        rating = 4
    elif 15 <= mpg < 17:
        rating = 3
    elif 14 <= mpg < 15:
        rating = 2
    else:
        rating = 1
    return rating


def randomize_in_place(alist, parallel_list=None, seed=None):
    """
    Randomize the order of the elements in alist.
    If parallel_list is not None, then it is also randomized in the same way while being kept parallel.
    """
    if seed is not None:
        np.random.seed(seed)

    for i in range(len(alist)):
        rand_index = np.random.randint(0, len(alist))  # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
    if parallel_list is not None:
        return alist, parallel_list
    else:
        return alist


def error_rate(predicted_labels, actual_labels):
    """
    Computes the error rate given predicted labels and actual labels.
    multiclass
    """
    num_errors = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != actual_labels[i]:
            num_errors += 1
    return num_errors / len(predicted_labels)


def folds_to_train_test(X_train_folds, X_test_folds, y_train_folds, y_test_folds):

    # now with X_train_folds, we create X_train
    X_train = []
    for fold in X_train_folds:
        for instance in fold:
            X_train.append(instance)
    X_test = []
    for fold in X_test_folds:
        for instance in fold:
            X_test.append(instance)
    y_test = []
    for fold in y_test_folds:
        for instance in fold:
            y_test.append(instance)
    y_train = []
    for fold in y_train_folds:
        for instance in fold:
            y_train.append(instance)
    return X_train, X_test, y_train, y_test


def indexes_to_fold(X_train_indexes, X_test_indexes, X, y):
    # now find y_train and y_test
    y_train_folds = []
    y_test_folds = []
    for test in X_test_indexes:
        index_y_test = []
        for index in test:
            index_y_test.append(y[index])
        y_test_folds.append(index_y_test)

    for train in X_train_indexes:
        index_y_train = []
        for index in train:
            index_y_train.append(y[index])
        y_train_folds.append(index_y_train)
    X_test_folds = []
    for fold in X_test_indexes:
        index_X_test = []
        for index in fold:
            index_X_test.append(X[index])
        X_test_folds.append(index_X_test)

    X_train_folds = []
    for fold in X_train_indexes:
        index_X_train = []
        for index in fold:
            index_X_train.append(X[index])
        X_train_folds.append(index_X_train)
    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def stratify(y, folds, random_state=None):
    """
    Stratifies the given labels into the given number of folds.
    """
    # first we need to find the number of unique labels
    unique_labels = set(y)
    # now we create a list of lists, where each list is the indexes of the instances with the same label
    index_lists = []
    for label in unique_labels:
        index_lists.append([i for i, x in enumerate(y) if x == label])
    # now we need to randomize the indexes for each label
    for index_list in index_lists:
        randomize_in_place(index_list, seed=random_state)
    # now we need to split the indexes into the folds
    index_lists_folds = []
    for i in range(folds):
        index_lists_folds.append([])
    for index_list in index_lists:
        for i in range(folds):
            index_lists_folds[i].append(
                index_list[i*(len(index_list)//folds):(i+1)*(len(index_list)//folds)])
    return index_lists_folds


def stratify_in_place(X, y):
    """
    Stratifies the given labels into the given number of folds.

    REturns: X,y
    """
    # first we need to find the number of unique labels
    unique_labels = set(y)
    # now we create a list of lists, where each list is the indexes of the instances with the same label
    index_lists = []
    for label in unique_labels:
        index_lists.append([i for i, x in enumerate(y) if x == label])
    # now we need to randomize the indexes for each label
    for index_list in index_lists:
        randomize_in_place(index_list)
    # now we need to split the indexes into the folds
    index_lists_folds = []
    for i in range(len(index_lists[0])):
        index_lists_folds.append([])
    for index_list in index_lists:
        for i in range(len(index_lists[0])):
            index_lists_folds[i].append(
                index_list[i*(len(index_list)//len(index_lists)):(i+1)*(len(index_list)//len(index_lists))])
    X_folds = []
    y_folds = []
    for fold in index_lists_folds:
        X_fold = []
        y_fold = []
        for index_list in fold:
            for index in index_list:
                X_fold.append(X[index])
                y_fold.append(y[index])
        X_folds.append(X_fold)
        y_folds.append(y_fold)
    return X_folds, y_folds


# def stratified_kfold_split(X, y, n_splits):
#     """
#     Stratified k-fold split.
#     """
#     # first we need to find the number of unique labels
#     unique_labels = set(y)
#     # now we create a list of lists, where each list is the indexes of the instances with the same label
#     index_lists = []
#     for label in unique_labels:
#         index_lists.append([i for i, x in enumerate(y) if x == label])
#     # now we need to randomize the indexes for each label
#     for index_list in index_lists:
#         randomize_in_place(index_list)
#     # now we need to split the indexes into the folds
#     index_lists_folds = []
#     for i in range(n_splits):
#         index_lists_folds.append([])
#     for index_list in index_lists:
#         for i in range(n_splits):
#             index_lists_folds[i].append(
#                 index_list[i*(len(index_list)//n_splits):(i+1)*(len(index_list)//n_splits)])
#     return index_lists_folds


def group_by(data):
    # groupby_col_index = header.index(groupby_col_name)  # use this later
    group_names = sorted(list(set(data)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]

    for i in range(len(data)):
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(data[i])
        group_subtables[groupby_val_subtable_index].append(i)

    return group_names, group_subtables


def get_column(table, col_index):
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col


# niave bayes
def p_value(X_train, new_instance, y_train, value):
    """
    Calculates the probability of a value given a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute.
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    float
        The probability of a value given a training set
    """
    p_X = p_list(X_train, new_instance, y_train, value)
    p_X_Mult = multiply_list(p_X)
    p_yes = p_X_Mult * y_train.count(value)
    return p_yes


def multiply_list(data):
    """
    Multiplies the values in a list

    Parameters
    ----------
    data : list
        A list of floats.

    Returns
    -------
    float
        The product of the values in the list
    """
    product = 1
    for i in range(len(data)):
        product *= data[i]
    return product


def p_list(X_train, new_instance, y_train, value):
    """
    Calculates the probability of each attribute in a training set

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    new_instance : list
        A list of strings. Each string represents an attribute. (X_train[i])
    y_train : list
        A list of strings. Each string represents a class label.
    value : string
        A string representing a class label.

    Returns
    -------
    list
        A list of floats. Each float represents the probability of an
        attribute in a training set.
    """
    perdiction_list = []
    for i in range(len(new_instance)):
        attribute_matches_count = 0
        for j in range(len(X_train)):
            if (new_instance[i] == X_train[j][i]):
                if (y_train[j] == value):
                    attribute_matches_count += 1
        perdiction_list.append(
            attribute_matches_count/y_train.count(value))
    return perdiction_list


def gaussian(x, mean, sdev):
    """
    Calculates the probability of a value given a training set

    Parameters
    ----------
    x : float
        A float representing an attribute.
    mean : float
        A float representing the mean of a distribution.
    sdev : float
        A float representing the standard deviation of a distribution.

    Returns
    -------
    float
        The probability of a value given a training set
    """
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second


def get_priors(y_train):
    """
    Calculates the prior probability of each class label

    Parameters
    ----------
    y_train : list
        A list of strings. Each string represents a class label.

    Returns
    -------
    Dict
        A dictionary where the keys are class labels and the values are
        prior probabilities.
    """
    unique_labels = set(y_train)
    priors = {}
    for label in unique_labels:
        priors[label] = y_train.count(label)/len(y_train)
    return priors


def get_posteriors(X_train, y_train, priors):
    """
    Calculates the posterior probability of each class label

    Parameters
    ----------
    X_train : list
        A list of lists of strings. Each inner list represents a training
        instance.
    y_train : list
        A list of strings. Each string represents a class label.
    priors : list
        A list of floats. Each float represents the prior probability of a
        class label.

    Returns
    -------
    Dict
        A dictionary where the keys are class labels and the values are
        posterior probabilities.
    """
    unique_labels = set(y_train)
    posteriors = {}
    for label in unique_labels:
        posteriors[label] = p_value(
            X_train, X_train[0], y_train, label) * priors[label]
        # this is a really gross number so we will round it to be 5 decimal places
        posteriors[label] = round(posteriors[label], 6)
    return posteriors


def get_prediction_naive_bayes(row, posteriors, priors):
    """
    Calculates the prediction of a row using naive bayes

    Parameters
    ----------
    posteriors : Dict
        A dictionary where the keys are class labels and the values are
        posterior probabilities.

    Returns
    -------
    string
        A string representing the predicted class label.
    """
    max_posterior = 0
    prediction = ""
    for label in posteriors:
        if posteriors[label] > max_posterior:
            max_posterior = posteriors[label]
            prediction = label
    return prediction


# TREEEEEEE

def tdidt(current_instances, available_attributes, header):
    # basic approach (uses recursion!!):
    # print(current_instances)
    # select an attribute to split on
    attribute = select_attribute(
        current_instances, available_attributes, header)

    available_attributes.remove(attribute)
    # this subtree
    tree = ["Attribute", attribute]
    partitions = partition_instances(
        current_instances, attribute, header, available_attributes)
    # group data by attribute domains (creates pairwise disjoint partitions)
    #   this is a grouopby where you use the attribute domain, instead of the values

    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():  # dictionary
        # print("Current att_value", att_value)
        # print("Length of partition", len(att_partition))
        value_subtree = ["Value", att_value]

        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            # print("CASE 1 all same class")
            # make leaf node)
            total_in_this_node = len(att_partition)
            total_include_others = len(current_instances)
            node = ["Leaf", att_partition[0][-1],
                    total_in_this_node, total_include_others]
            value_subtree.append(node)

        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2 no more attributes")
            value_subtree.append(
                ["Leaf", majority_vote(att_partition), len(att_partition), len(current_instances)])

            # handle clash with the majority vote leaf node

        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            # print("CASE 3 empty partition")
            # backtrack and replace this attribute node with a majority vote leaf node
            value_subtree.append(
                ["Leaf", majority_vote(current_instances), len(current_instances), len(current_instances)])
        else:
            # none of the previous conditions were true...
            # recurse :)
            # need a .copy here because we cant split on the same attribute twice
            # print("Woah recursion")
            subtree = tdidt(att_partition, available_attributes.copy(), header)
            # now that we have this subtree:
            # append subtree to value_subtree, and then tree appropriatly
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree


def majority_vote(instances):
    # get the class labels
    class_labels = [instance[-1] for instance in instances]
    # get the unique class labels
    unique_labels = set(class_labels)
    max_count = 0
    for instance in instances:
        # get the class label
        label = instance[-1]
        # get the count of the class label
        count = class_labels.count(label)
        # if the count is greater than the max count, set the max count to the count
        if count > max_count:
            max_count = count
            max_label = label
    return max_label


def all_same_class(att_partition):
    # look through all the [-1] and if they all are the same return true
    for i in range(len(att_partition)):
        try:
            if att_partition[i][-1] == att_partition[i+1][-1]:
                continue
            else:
                return False
        except IndexError:
            return True


def select_attribute(current_instances, available_attributes, header):
    # TODO:
    # use entropy to calculate and chose the attribute
    # with the smallest E_new

    # for now we will use random attribute selection
    # rand_index = np.random.randint(0, len(available_attrbutes))

    # * for each available attribute:
    #     * for each value in the attribute's domain (Seinor, Junior, etc...)
    #         * calculate the entropy of that value's partition (E_Seinor, E_Junior, etc...)
    #     * computer E_new, which is the weighted sum of the partition entropies
    # * chose to split on the attribute with the smallest E_new

    # print("The random index chosen is", rand_index)
    smallest_E_new = 100000
    for attribute in available_attributes:
        partitians = partition_instances(
            current_instances, attribute, header, available_attributes)
        partitian_entropies = get_partition_entropies(partitians)
        e_new = get_e_new(partitian_entropies, partitians)
        if e_new < smallest_E_new and len(partitian_entropies) > 0:
            smallest_E_new = e_new
            best_attribute = attribute

    return best_attribute
    # return available_attributes[index]


def get_e_new(partitian_entropies, partitians):
    e_new = 0
    total = 0
    e_vals = []
    for partitian in partitians:
        partitian_total = 0
        for group in partitians[partitian]:
            partitian_total += 1
            total += 1
        e_vals.append(partitian_total)
    for e_val in e_vals:
        e_new += (e_val/total) * partitian_entropies[e_vals.index(e_val)]
    return e_new


def get_partition_entropies(partitions):
    # get the entropy for each partition
    entropies = []
    for partition in partitions.values():
        # get the entropy for this partition
        entropy = get_entropy(partition)
        # append the entropy to the entropies list
        entropies.append(entropy)
    return entropies


def get_entropy(partition):
    # get the class labels
    class_labels = [instance[-1] for instance in partition]
    # get the unique class labels
    unique_labels = set(class_labels)
    # get the number of instances in each class
    class_counts = [class_labels.count(label) for label in unique_labels]
    # get the total number of instances
    total_count = len(partition)
    # get the probability of each class
    probabilities = [count/total_count for count in class_counts]
    # get the entropy for this partition
    entropy = get_entropy_from_probabilities(probabilities)
    return entropy


def get_entropy_from_probabilities(probabilities):
    # get the entropy from the probabilities
    # THIS IS ASSUMONG ITS NOT 0
    entropy = -sum([prob * math.log(prob, 2) for prob in probabilities])
    return entropy


def partition_instances(current_instances, split_attribute, header, attribute_domains):
    # group by attribute domain
    #   use the attrobite_domains thing
    # key (attribute value)[junior, mid, seinor]: value (subtable) [values for junior, values for mid, values for senior]
    partitions = {}
    att_index = header.index(split_attribute)  # e.g. 0
    # att_domain = attribute_domains[att_index]  # e.g. ["Junior","Senior","Mid"]
    att_domain = dict()
    for i in range(len(current_instances)):
        if current_instances[i][att_index] not in att_domain:
            att_domain[current_instances[i][att_index]] = []
        att_domain[current_instances[i][att_index]].append(
            current_instances[i])
    for att_value in att_domain:
        # make an empty list at the key of att_value (Junior,Seinor, etc...)
        partitions[att_value] = []
        for instance in current_instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    # return a dictionary
    return partitions


def tdidt_predict(tree, instance):
    # recursively traverse the tree
    # we need to know wher we are in the tree
    # are we at a leaf node? (base case)
    # if not, we are at an attribute node
    #
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    # we dont have a base case, so we need to recurse
    # we need to amtch the attribute's value in the instance
    # with the appropriate value list
    # a for loop that traverses through each
    # value list
    # recurse on match with instance value
    att_index = int(tree[1][3])  # get the number after att

    for i in range(2, len(tree)):
        value_list = tree[i]
        # checking if the value of the instance is in the value list
        if value_list[1] == instance[att_index]:
            return tdidt_predict(value_list[2], instance)


# def get_line_rule(tree, attribute_names=None, class_label="class", depth=0):
#     # now its either attribute or value
#     uniques = []
#     for offset in range(len(tree)-2):
#         print(len(tree))
#         print("offset", offset)
#         if tree[0] == "Leaf":  # reahed the end of a rule
#             return "THEN " + class_label + " = " + tree[1]
#         if tree[0] == "Attribute":
#             if attribute_names == None:
#                 if depth == 0:
#                     uniques.append("IF " + tree[1] + " = " + get_line_rule(
#                         tree[2+offset], attribute_names, class_label, depth+1))
#                 else:
#                     return " AND " + tree[1] + " = " + get_line_rule(tree[2+offset], attribute_names, class_label, depth+1)
#             else:
#                 return attribute_names[int(tree[1][2:])] + " = " + get_line_rule(tree[2+offset], attribute_names, class_label, depth+1)
#         elif tree[0] == "Value":
#             return str(tree[1]) + " " + get_line_rule(tree[2+offset], attribute_names, class_label, depth+1)
#     return uniques

def get_lines_rules_encoded(tree, attribute_names=None, class_label="class", depth=0):
    # now its either attribute or value
    uniques = []
    if tree[0] == "Leaf":  # reahed the end of a rule
        return "THEN " + class_label + " = " + tree[1]

    if tree[0] == "Attribute":
        if attribute_names == None:
            routes = []
            for i in range(2, len(tree)):
                if depth == 0:
                    routes.append("IF " + tree[1] + " = ")
                    routes.append(get_lines_rules_encoded(
                        tree[i], attribute_names, class_label, depth+1))
                else:
                    routes.append(" AND " + tree[1] + " = ")
                    routes.append(get_lines_rules_encoded(
                        tree[i], attribute_names, class_label, depth+1))
            return routes
        else:
            routes = []
            for i in range(2, len(tree)):
                if depth == 0:
                    routes.append(
                        "IF " + attribute_names[int(tree[1][3:])] + " = ")
                    routes.append(get_lines_rules_encoded(
                        tree[i], attribute_names, class_label, depth+1))
                else:
                    routes.append(
                        " AND " + attribute_names[int(tree[1][3:])] + " = ")
                    routes.append(get_lines_rules_encoded(
                        tree[i], attribute_names, class_label, depth+1))
            return routes

    if tree[0] == "Value":
        routes = []
        for i in range(2, len(tree)):
            routes.append(str(tree[1]))
            routes.append(get_lines_rules_encoded(
                tree[i], attribute_names, class_label, depth+1))
        return routes
    return routes


def get_all_line_rules(tree, attribute_names=None, class_label="class"):
    routes = get_lines_rules_encoded(tree, attribute_names, class_label)
    decoded_routes = []
    decode_line(routes, decoded_routes)
    new_rules = remove_invalid_rules(decoded_routes)
    return new_rules


def decode_line(route_pos, decoded_routes, previous_string=None):
    if previous_string is None:
        line = str(route_pos[0])
        for i in range(1, len(route_pos)):
            decoded_routes.append(decode_line(
                route_pos[i], decoded_routes, line))
    else:
        if type(route_pos) == list:
            line = previous_string + str(route_pos[0])
            for i in range(1, len(route_pos)):
                # line += str(decode_line(route_pos[i], decoded_routes, line))
                decoded_routes.append(
                    str(decode_line(route_pos[i], decoded_routes, line)))
        else:
            line = previous_string + " " + str(route_pos)
            return line


def remove_invalid_rules(rules):
    while("None" in rules):
        rules.remove("None")
    while(None in rules):
        rules.remove(None)
    remove_incomplete_rules(rules)
    return rules


def remove_incomplete_rules(rules):
    for rule in rules:
        if "THEN" not in rule:
            rules.remove(rule)
