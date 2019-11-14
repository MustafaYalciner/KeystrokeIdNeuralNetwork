import tensorflow as tf
import pandas as p

class Main:
    if __name__ == '__main__':
        data = p.read_csv('data/featureset.csv', nrows=20)
        print(data.head(10))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
