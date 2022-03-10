import tensorflow as tf
import collections
import matplotlib.pyplot as plt

def get_data():
    """
     get the dataset that will be used
    :return: the training and testing set
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    y_count = collections.Counter(y_train) # dictionary label:number of samples
    keylist= sorted(y_count)
    y_count_sorted = [y_count[k] for k in keylist]
    plt.bar(keylist, y_count_sorted)
    plt.title("training_data_description")
    plt.yticks(y_count_sorted)
    plt.ylabel("nbr of sample")
    plt.legend("Training data size: " + str(len(x_train)) , loc='upper center')
    plt.savefig("datasets/training_data_description.png")

    return (x_train, y_train), (x_test, y_test)