##############################################
# Programmers: Sam Berkson and Ben Puryear
# Class: CPSC 322-02, Spring 2022
# Final Project
# 5/4/22
# 
# Description: This is the file that contains the functions that will be used to
# plot the data.
##############################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams["savefig.directory"] = os.path.dirname(os.path.abspath(__file__)) + "/media"



def plot_occurance_bar(mypytable, attribute, title=None, limit=None, rotation=0):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    attribute_col = mypytable.get_column(attribute)
    if limit is not None:
        attribute_col = attribute_col[:limit]
    no_dupes = list(set(attribute_col))
    occurance = [attribute_col.count(x) for x in no_dupes]
    plt.bar(no_dupes, occurance)
    if title is not None:
        plt.title(title)
    else:
        title = "Occurance Chart"
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")

    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_percentage_of_total_pie(mypytable, attributes, title=None, limit=None):
    """
    Plot a pie chart of the percentage of multiple attributes
    """
    attribute_vals = [mypytable.get_column(x) for x in attributes]

    if limit is not None:
        attribute_vals = [x[:limit] for x in attribute_vals]
    attribute_sums = []
    for attribute in attribute_vals:
        attribute_total = 0
        for val in attribute:
            try:
                val = float(val)
            except ValueError:
                continue
            attribute_total += val
        attribute_sums.append(attribute_total)
    plt.pie(attribute_sums, labels=attributes, autopct="%1.1f%%")
    if title is not None:
        plt.title(title)
    else:
        title = "Pie Chart"

    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_occurance_bar_list(
    data, attribute=None, title=None, limit=None, rotation=0, x_labels=[]
):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    no_dupes = list(set(data))
    occurance = [data.count(x) for x in no_dupes]
    plt.bar(no_dupes, occurance)
    if title is not None:
        plt.title(title)
    else:
        title = "Bar Plot"
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(x_labels, rotation=rotation)
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_histogram(data, attribute=None, title=None, limit=None, rotation=0):
    """
    Plot a histogram of the occurance of a single attribute
    """
    # 10 bins
    plt.hist(data, bins=10)
    if title is not None:
        plt.title(title)
    else:
        title = "Histogram"
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_scatter(data, attribute_x, attribute_y, title=None, limit=None, rotation=0):
    """
    Plot a scatter plot of the occurance of a single attribute
    """
    plt.scatter(data.get_column(attribute_x), data.get_column(attribute_y))
    if title is not None:
        plt.title(title)
    else:
        title = "Scatter Plot"
    plt.xlabel(attribute_x)
    plt.ylabel(attribute_y)
    plt.xticks(rotation=rotation)
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_scatter_list(x, y, title=None, limit=None, rotation=0):
    """
    Plot a scatter plot of the occurance of a single attribute
    """
    plt.scatter(x, y)
    if title is not None:
        plt.title(title)
    else:
        title = "Scatter Plot"
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.xticks(rotation=rotation)
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_bar_dict(dictionary, title=None, limit=None, rotation=0):
    """
    Plot a bar chart of the occurance of a single attribute
    """
    plt.bar(dictionary.keys(), dictionary.values())
    if title is not None:
        plt.title(title)
    else:
        title = "Bar Plot"
    plt.xlabel("Attribute")
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()


def plot_box_and_whisker(data, attribute=None, title=None, limit=None, rotation=0):
    """
    Plot a box and whisker plot of an array
    """
    plt.boxplot(data)
    if title is not None:
        plt.title(title)
    else:
        title = "Box and Whisker Plot"
    plt.xlabel(attribute)
    plt.ylabel("Occurance")
    plt.xticks(rotation=rotation)
    for i in title:
        if i == " ":
            title = title.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()

def hist_helper(column, name, xlabel, ylabel):
    """
    Helper function for plotting histograms
    """
    plt.figure()
    plt.hist(column, align = 'mid', bins = 10, color = 'blue', edgecolor = 'black', linewidth = 1.2)
    plt.xticks(rotation = 90)
    plt.xlabel(xlabel, color = 'black')
    plt.ylabel(ylabel, color = 'black')
    plt.title(name, color = 'black')
    plt.tick_params(axis = 'x', colors = 'black')
    plt.tick_params(axis = 'y', colors = 'black')
    for i in name:
        if i == " ":
            name = name.replace(" ", "_")
    plt.savefig("media/" + name + ".png")
    plt.show()

def pie_chart_helper(data, label, title):
    """
    Helper function for plotting pie charts
    """
    plt.figure()
    plt.pie(data, labels = label, autopct = '%1.1f%%', startangle = 90, shadow = True, textprops={'color':'pink'}, normalize=True)
    plt.title(title, color = 'black')
    for i in name:
        if i == " ":
            name = name.replace(" ", "_")
    plt.savefig("media/" + title + ".png")
    plt.show()

def bar_chart_helper(x, y, name, xlabel, ylabel):
    '''
    Helper function for plotting bar charts
    '''
    plt.figure()
    plt.bar(x, y, color = 'blue' , edgecolor = 'black', linewidth = 1.1)
    plt.title(name, color = 'black')
    plt.xlabel(xlabel, color = 'black')
    plt.ylabel(ylabel, color = 'black')
    plt.xticks(rotation=90, color = 'black')
    plt.yticks(color = 'black')
    plt.tick_params(axis = 'x', color = 'black')
    plt.tick_params(axis = 'y', color = 'black')
    for i in name:
        if i == " ":
            name = name.replace(" ", "_")
    plt.savefig("media/" + name + ".png")
    plt.show()

def scatter_plot_helper(x, y, name, xlabel, ylabel, regression = True):
    '''
    Helper function for plotting scatter plots
    '''
    plt.figure()
    plt.scatter(x, y, color = 'blue', edgecolor = 'black', linewidth = 1.2)
    plt.title(name, color = 'black')
    plt.xlabel(xlabel, color = 'black')
    plt.ylabel(ylabel, color = 'black')
    plt.tick_params(axis = 'x', color = 'black')
    plt.tick_params(axis = 'y', color = 'black')
    if regression:
        m, b = compute_slope_intercept(x, y)
        print("Slope = " + str(m))
        plt.plot([min(x), max(x)],[m*min(x) + b, m*max(x) + b],c="r",lw=5)
    plt.xticks(color = 'black')
    plt.yticks(color = 'black')
    plt.grid(True)
    for i in name:
        if i == " ":
            name = name.replace(" ", "_")
    plt.savefig("media/" + name + ".png")
    plt.show()

def compute_slope_intercept(x, y):
    '''
    Helper function to calculate slope and intercept for plotting of least squares linear regression
    '''
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = 0
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
    
    denominator = 0
    for i in range(len(x)):
        denominator += (x[i] - x_mean) ** 2
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept