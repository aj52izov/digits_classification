import tensorflow as tf
import tensorflow.keras as keras
import preprocessing.preprocessing as preprocessing
import datasets.dataset as dts
import os
import matplotlib.pyplot as plt


def train(model, data, save_directory="", batch_size=32, epochs=50, optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(data, batch_size=batch_size, epochs=epochs)
    if save_directory != "":
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    model.save(save_directory + "/model.h5")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.savefig(save_directory + "/training_curve.png")

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = dts.get_data()
    x_train = preprocessing.apply(x_train)
    x_test = preprocessing.apply(x_test)