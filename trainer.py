import tensorflow as tf
import model.model as ml
import preprocessing.preprocessing as preprocessing
import datasets.dataset as dts
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def train(model, data, save_directory="trained_model", batch_size=32, epochs=30, optimizer="adam",
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
    history = model.fit(data[0], data[1], batch_size=batch_size, epochs=epochs, validation_split=0.2)
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    # save trained model
    model.save(save_directory + "/model.h5")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.ylim(top=1)
    plt.legend(['train_accuracy', 'train_loss', 'val_accuracy', 'val_loss'], loc='upper left')
    plt.savefig(save_directory + "/training_curves.png")
    plt.close()
    return model


def evaluate(model, data, save_directory="trained_model", class_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    evaluate the model on a data set and save the confusion matrix

    :param model: the model to evaluated
    :param data: the validation dataset
    :param save_directory: the directory where the confusion matrix will be saved
    :param class_names: the class map corresponding to the model outputs
    :return: the validation loss a accuracy
    """

    val_loss, val_acc = model.evaluate(data[0], data[1])
    y_test = data[1]
    predict = model.predict(data[0])
    y_pred = tf.argmax(predict,1) # get predicted classes
    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
    ind = np.arange(len(class_names))

    plt.figure(figsize=(15, 10))
    sns.heatmap(confusion_mtx / np.sum(confusion_mtx), annot=True, fmt='.2%', cmap='Blues')  # font size

    plt.title('Confusion_Matrix: nbr of sample={}, acc={}, loss={}'.format(len(y_test), round(val_acc, 3),round(val_loss, 3)))
    plt.xlabel('Predicted value')
    plt.ylabel('True value')
    plt.xticks(ind+0.5, class_names)
    plt.yticks(ind+0.5, class_names)
    plt.savefig(save_directory + "/Confusion_Matrix.png")
    plt.close()
    return val_acc , val_loss


if __name__ == "__main__":
    # load data
    print("Loadind data ")
    (x_train, y_train), (x_test, y_test) = dts.get_data()

    # apply preprocessing
    print("Applying filtering on data ")
    x_train = preprocessing.apply(x_train)
    x_test = preprocessing.apply(x_test)

    # get model pipeline
    print("getting model pipeline ")
    model = ml.make_pipeline()

    # Train the model
    print(" Training the model")
    model = train(model, (x_train, y_train))

    #evaluate the model
    print(" evaluating the model")
    #model = tf.keras.models.load_model("model/my_model.h5")
    val_acc, val_loss = evaluate(model, (x_test, y_test))

