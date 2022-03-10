import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pdb


def make_pipeline(input_shape=(28, 28)):
    """
    define the model layer-by-layer using the keras abstraction and save the defined model graph in model_graph.png
    :param input_shape: the shape of the input images
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
    keras.utils.plot_model(model, to_file='model_graph.png', show_shapes=True, show_layer_names=True)
    return model


#model = make_pipeline()
#print(model.summary())

#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy']
#              )

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#y_count = collections.Counter(y_train)
#keylist = sorted(y_count)
#y_count_sorted = [y_count[k] for k in keylist]
#print(y_count)
#plt.bar(keylist, y_count_sorted)
#plt.show()

#pdb.set_trace()
#x_train = preprocessing.apply(x_train)
#x_test = preprocessing.apply(x_test)

#history = model.fit(x_train, y_train, batch_size=None, epochs=50)

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['loss'])
#plt.legend(['accuracy', 'loss'], loc='upper left')
#plt.show()


#model.save("my_model.h5")
