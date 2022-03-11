import tensorflow as tf
import collections
import matplotlib.pyplot as plt
import numpy as np
def get_data():
    """
     get the dataset that will be used
    :return: the training and testing set
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    y_count = collections.Counter(y_train) # dictionary label:number of samples
    keylist= sorted(y_count)
    y_count_sorted = [y_count[k] for k in keylist]
    ind = np.arange(len(keylist))    # the x locations for the groups

    fig, ax = plt.subplots()
    ax.bar(ind, y_count_sorted)
    ax.set_xticks(ind)
    ax.set_yticklabels(keylist)
    ax.set_title("Training_data_description sample={}".format(len(x_train)))
    ax.set_ylabel('nbr of sample')
    ax.set_xlabel('classes')
    ax.bar_label(ax.containers[0])
    plt.savefig("datasets/training_data_description.png")
    plt.close()
    return (x_train, y_train), (x_test, y_test)