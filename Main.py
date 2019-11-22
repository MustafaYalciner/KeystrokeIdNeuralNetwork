import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Main:
    if __name__ == '__main__':
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        data = pd.read_csv('data/featureset.csv', nrows=100000)
        data = data.drop(columns=["DATASET", "SENTENCE_ID"])
        train, test = train_test_split(data, test_size=0.1)
        train_sol = train[["USER_ID"]]
        test_sol = test[["USER_ID"]]
        print("Number of unique users:")
        print(test_sol['USER_ID'].nunique())
        # encode class values as integers
        print(train_sol)
        encoder = LabelEncoder()
        encoder.fit(train_sol)
        encoded_Y = encoder.transform(train_sol)
        # convert integers to dummy variables (i.e. one hot encoded)
        train_sol = tf.keras.utils.to_categorical(encoded_Y)

        encoder = LabelEncoder()
        encoder.fit(test_sol)
        encoded_Y2 = encoder.transform(test_sol)
        print(encoded_Y2)
        # convert integers to dummy variables (i.e. one hot encoded)
        test_sol = tf.keras.utils.to_categorical(encoded_Y2)
        #print(numpy.train_sol.nunique())
        print(train_sol)
        print(type(train_sol))

        train_data = train.drop(columns=["USER_ID"])
        test_data = test.drop(columns=["USER_ID"])
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(26))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_data, train_sol, validation_split=0.2, epochs=10)
        test_prediction = model.predict(test_data)
        print(test_prediction)
        print(type(test_prediction))
        print(test_sol)
        print(type(test_sol))
