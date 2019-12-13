import tensorflow as tf
import pandas as pd
import random
import numpy as np
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier


class Main:

    if __name__ == '__main__':
        usersToTrainLabelEncoder = pd.read_csv('data/featureset.csv', usecols=['USER_ID'])
        print(usersToTrainLabelEncoder)

        ROW_COUNT = 4842367
        ALL_INDICES = [i for i in range(ROW_COUNT)]
        random.shuffle(ALL_INDICES)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        data = pd.read_csv('data/output.csv', nrows=200000)
        data = data.drop(columns=["DATASET", "SENTENCE_ID"])
        train, test = train_test_split(data, test_size=0.01, shuffle=True)

        train_current_key_code=pd.get_dummies(train['KEYCODE'], prefix='current')
        train_prev_current_key_code= pd.get_dummies(train['KEYCODE_PREV'], prefix='prev')
        train_tri_current_key_code= pd.get_dummies(train['KEYCODE_TRI'], prefix='tri')

        train = train.drop(columns=["KEYCODE", "KEYCODE_PREV", "KEYCODE_TRI"])
        train = train.join(train_current_key_code)
        train = train.join(train_prev_current_key_code)
        train = train.join(train_tri_current_key_code)

        train_sol = train[["USER_ID"]]
        test_sol = test[["USER_ID"]]

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(usersToTrainLabelEncoder)
        # convert integers to dummy variables (i.e. one hot encoded)
        train_sol = pd.get_dummies(encoder.transform(train_sol))

        # convert integers to dummy variables (i.e. one hot encoded)
        test_sol = pd.get_dummies(encoder.transform(test_sol))

        train_data = train.drop(columns=["USER_ID"])
        test_data = test.drop(columns=["USER_ID"])

      #  mlp = MLPClassifier(hidden_layer_sizes=(train_data.shape[1], 10, 10, 10), solver='sgd')

        model = tf.keras.models.Sequential()
        # The batch size of this model is 10, even though the paper works with a batch size of 1.
        # Because when using a batch size of 1, the loss function returns NaN.
        model.add(tf.keras.layers.Dense(10, activation='relu', batch_input_shape=(1, train_data.shape[1])))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(53, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
#       mlp.fit(train_data,train_sol)
        model.fit(train_data, train_sol, validation_split=0.01, epochs=20)
        #test_prediction = model.predict(test_data, steps=1)

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.5, nesterov=False),
#categorical_accuracy: checks to see if the index of the maximal true value is equal to the index of the maximal predicted value.
#
#        model.add(tf.keras.layers.Dropout(0.5))
 #       model.add(tf.keras.layers.BatchNormalization())