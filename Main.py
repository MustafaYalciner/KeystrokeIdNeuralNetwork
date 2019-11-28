import tensorflow as tf
import pandas as pd
import random
import numpy as np
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Main:

    if __name__ == '__main__':
        ROW_COUNT = 4842367
        ALL_INDICES = [i for i in range(ROW_COUNT)]
        random.shuffle(ALL_INDICES)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        data = pd.read_csv('data/featureset.csv', nrows=100000)
        data = data.drop(columns=["DATASET", "SENTENCE_ID"])
        train, test = train_test_split(data, test_size=0.01)
        train_sol = train[["USER_ID"]]
        test_sol = test[["USER_ID"]]

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(train_sol)
        # convert integers to dummy variables (i.e. one hot encoded)
        train_sol = tf.keras.utils.to_categorical(encoder.transform(train_sol))

        encoder = LabelEncoder()
        encoder.fit(test_sol)
        # convert integers to dummy variables (i.e. one hot encoded)
        test_sol = tf.keras.utils.to_categorical(encoder.transform(test_sol))

        train_data = train.drop(columns=["USER_ID"])
        test_data = test.drop(columns=["USER_ID"])
        model = tf.keras.models.Sequential()

        # The batch size of this model is 10, even though the paper works with a batch size of 1.
        # When using a batch size of 1, however, the loss function returns NaN.
        model.add(tf.keras.layers.Dense(10, activation='relu', batch_input_shape=(10, train_data.shape[1])))
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
        model.add(tf.keras.layers.Dense(26))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.fit(train_data, train_sol, validation_split=0.2, epochs=60)
        test_prediction = model.predict(test_data, steps=1)


#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.5, nesterov=False),
#categorical_accuracy: checks to see if the index of the maximal true value is equal to the index of the maximal predicted value.
#
#        model.add(tf.keras.layers.Dropout(0.5))
 #       model.add(tf.keras.layers.BatchNormalization())