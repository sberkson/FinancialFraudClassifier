# Sam Berkson
# CPSC 322
# PA3
# Plotting Utility Functions

import matplotlib.pyplot as plt

def hist_helper(column, name, xlabel, ylabel):
    """
    Helper function for plotting histograms
    """
    plt.figure()
    plt.hist(column, align = 'mid', bins = 10, color = 'blue', edgecolor = 'black', linewidth = 1.2)
    plt.xticks(rotation = 90)
    plt.xlabel(xlabel, color = 'red')
    plt.ylabel(ylabel, color = 'red')
    plt.title(name, color = 'red')
    plt.tick_params(axis = 'x', colors = 'red')
    plt.tick_params(axis = 'y', colors = 'red')

def pie_chart_helper(data, label, title):
    """
    Helper function for plotting pie charts
    """
    plt.figure()
    plt.pie(data, labels = label, autopct = '%1.1f%%', startangle = 90, shadow = True, textprops={'color':'pink'}, normalize=True)
    plt.title(title, color = 'red')
    plt.show()

def bar_chart_helper(x, y, name, xlabel, ylabel):
    '''
    Helper function for plotting bar charts
    '''
    plt.figure()
    plt.bar(x, y, color = 'blue' , edgecolor = 'black', linewidth = 1.1)
    plt.title(name, color = 'red')
    plt.xlabel(xlabel, color = 'red')
    plt.ylabel(ylabel, color = 'red')
    plt.xticks(rotation=90, color = 'red')
    plt.yticks(color = 'red')
    plt.tick_params(axis = 'x', color = 'red')
    plt.tick_params(axis = 'y', color = 'red')
    plt.show()

def scatter_plot_helper(x, y, name, xlabel, ylabel, regression = True):
    '''
    Helper function for plotting scatter plots
    '''
    plt.figure()
    plt.scatter(x, y, color = 'blue', edgecolor = 'black', linewidth = 1.2)
    plt.title(name, color = 'red')
    plt.xlabel(xlabel, color = 'red')
    plt.ylabel(ylabel, color = 'red')
    plt.tick_params(axis = 'x', color = 'red')
    plt.tick_params(axis = 'y', color = 'red')
    if regression:
        m, b = compute_slope_intercept(x, y)
        plt.plot([min(x), max(x)],[m*min(x) + b, m*max(x) + b],c="r",lw=5)
    plt.xticks(color = 'red')
    plt.yticks(color = 'crimson')
    plt.grid(True)
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