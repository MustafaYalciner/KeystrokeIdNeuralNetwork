import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SlicedDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, trainedEncoder, shuffled_indexes, labels, batch_size=10, dim=(10, 15), n_channels=1,
                 n_classes=26, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.shuffled_indexes = shuffled_indexes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.encoder = trainedEncoder

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.shuffled_Indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        shuffled_indexes_batch = [self.shuffled_Indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(shuffled_indexes_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, shuffled_indexes_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        rows_to_skip = set(self.shuffled_indexes) - set(shuffled_indexes_batch)
        data_batch = pd.read_csv('data/featureset.csv', skiprows=rows_to_skip)
        X =data_batch.drop(columns=["DATASET", "SENTENCE_ID"])
        y=data_batch[['USER_ID']]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)