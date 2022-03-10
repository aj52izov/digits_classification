import tensorflow as tf
import model.model as ml
import preprocessing.preprocessing as preprocessing
import datasets.dataset as dts
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools


def _plot_confusion_matrix(cm, class_names):
    """
    :param cm (array, shape = [n, n]): a confusion matrix of integer classes
    :param class_names (array, shape = [n]): String names of the integer classes
    :return: a matplotlib figure containing the plotted confusion matrix.
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure



def train(model, data, save_directory="trained_model", batch_size=32, epochs=50, optimizer="adam",
          loss="sparse_categorical_crossentropy", metrics=['accuracy']):
    """
    train a model, save the trained model and its performance evolution(losss and accuracy) curves
    :param model: the to be trained model
    :param data: the training data (X,Y)
    :param save_directory: the directory where the trained model will be saved
    :param batch_size: the trainingb batch size
    :param epochs: the number of training epoch
    :param optimizer: the optimizer to use
    :param loss: the loss function
    :param metrics: the evaluation metrics
    :return: the resulting trained model
    """

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(data, batch_size=batch_size, epochs=epochs, validation_split=0.2, workers=4)
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    # save trained model
    model.save(save_directory + "/model.h5")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_accuracy', 'train_loss', 'val_accuracy', 'val_loss'], loc='upper left')
    plt.savefig(save_directory + "/training_curves.png")

    return model


def evaluate(model, data):
    val_loss, val_acc = model.evaluate(data)

    y_test = data[1]
    y_pred = model.predict(data[0], workers=4)

    confusion_mtx  = tf.math.confusion_matrix(y_test, y_pred)


    print("loss-> ", val_loss, "\nacc-> ", val_acc)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = dts.get_data()
    x_train = preprocessing.apply(x_train)
    x_test = preprocessing.apply(x_test)
    model = ml.make_pipeline()
    model = train(model, (x_train, y_train))