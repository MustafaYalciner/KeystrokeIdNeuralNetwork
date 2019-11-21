import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Main:
    if __name__ == '__main__':
        data = pd.read_csv('data/featureset.csv', nrows=100000)
        print(type(data))
        data = data.drop(columns=["DATASET", "SENTENCE_ID"])
        train, test = train_test_split(data, test_size=0.1)
        train_sol = train[["USER_ID"]]

        # encode class values as integers
        print(train_sol)
        encoder = LabelEncoder()
        encoder.fit(train_sol)
        encoded_Y = encoder.transform(train_sol)
        print(encoded_Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        train_sol = tf.keras.utils.to_categorical(encoded_Y)

        print(train_sol)
        print(type(train_sol))

        train_data = train.drop(columns=["USER_ID"])
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(26))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_data, train_sol, validation_split=0.2, epochs=30)
