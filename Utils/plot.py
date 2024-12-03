import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_kernel(data1, data2):

    sns.kdeplot(data1, shade=True, color="blue", label='Data 1')
    sns.kdeplot(data2, shade=True, color="red", label='Data 2')

    plt.title('Multiple Density Plots')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim([0, 100])

    plt.show()


def plot_hist(data1, data2):

    plt.hist(data1, bins=len(data1), alpha=0.7, color='blue', label='Data 1')
    plt.hist(data2, bins=len(data2), alpha=0.7, color='red', label='Data 2')

    plt.title('Multiple Hist Plots')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.show()

