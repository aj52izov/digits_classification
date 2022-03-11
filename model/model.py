import tensorflow as tf
import tensorflow.keras as keras

def make_pipeline(input_shape=(28, 28), save_directory="trained_model"):
    """
    define the model layer-by-layer using the keras abstraction and save the defined model graph in model_graph.png
    :param input_shape: the shape of the input images
    :param save_directory: the directory where the trained model graph  will be saved
    :return: a keras sequential model
    """

    model = tf.keras.models.Sequential()
    shape_size = len(input_shape)
    if shape_size == 1:
        model.add(keras.Input(shape=(input_shape[0],)))
    elif shape_size == 2:
        model.add(keras.Input(shape=(input_shape[0], input_shape[1],)))
    else:
        model.add(keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2],)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # save the model  graph
    keras.utils.plot_model(model, to_file=save_directory + "/model_graph.png", show_shapes=True, show_layer_names=True)
    return model
