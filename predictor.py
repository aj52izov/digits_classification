import tensorflow as tf
import model.model as ml
import preprocessing.preprocessing as preprocessing
import datasets.dataset as dts

def predict(model_file, data, class_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    make prediction on data set and return de result

    :param model: the trained  model
    :param data: the dataset to get the output
    :param class_names: the class map corresponding to the model outputs
    :return: the validation loss a accuracy
    """
    model = tf.keras.models.load_model(model_file)
    predict = model.predict(data)
    y_pred = tf.argmax(predict, 1).numpy().tolist() # get predicted classes
    print(y_pred[:5])
    predicted_class_names = [class_names[i] for i in y_pred]
    print(predicted_class_names[:5])
    return predicted_class_names


if __name__ == "__main__":
    # load data
    print("Loadind data ")
    (x_train, y_train), (x_test, y_test) = dts.get_data()

    # make prediction
    predicted_class_names = predict("trained_model/model.h5", x_test)