import tensorflow as tf
import collections
import matplotlib.pyplot as plt

def get_data():
    '''
     get the dataset that will be used
    :return:
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_count = collections.Counter(y_train)
    keylist = sorted(y_count)
    y_count_sorted = [y_count[k] for k in keylist]

    plt.bar(keylist, y_count_sorted)
    plt.title("Training data size: " + str(len(x_train)))
    plt.show()
    return (x_train, y_train), (x_test, y_test)