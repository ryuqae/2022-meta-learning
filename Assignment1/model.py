import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.python.ops.numpy_ops import np_config
import sys

np_config.enable_numpy_behavior()

import numpy as np


class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, padding: str = "same"):
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding=self.padding
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2, 2))

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Prototypical_Network(Model):
    def __init__(self, w: int = 28, h: int = 28, c: int = 1):
        super(Prototypical_Network, self).__init__()
        self.w, self.h, self.c = w, h, c

        self.encoder = tf.keras.Sequential(
            [
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                layers.Flatten(),
            ]
        )

    def call(self, support, query):
        n_way = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]

        reshaped_s = tf.reshape(support, (n_way * n_support, self.w, self.h, self.c))
        reshaped_q = tf.reshape(query, (n_way * n_query, self.w, self.h, self.c))

        # Embeddings are in the shape of (n_support+n_query, 64)
        embeddings = self.encoder(tf.concat([reshaped_s, reshaped_q], axis=0))

        # Support prototypes are in the shape of (n_way, n_support, 64)
        s_prototypes = tf.reshape(
            embeddings[: n_way * n_support], [n_way, n_support, embeddings.shape[-1]]
        )

        # Find the average of prototypes for each class in n_way
        s_prototypes = tf.math.reduce_mean(s_prototypes, axis=1)

        # Query embeddings are the remainding embeddings
        q_embeddings = embeddings[n_way * n_support :]

        loss = 0.0
        acc = 0.0
        ############### Your code here ###################
        # TODO: finish implementing this method.
        # For a given task, calculate the Euclidean distance
        # for each query embedding and support prototypes.
        # Then, use these distances to calculate
        # both the loss and the accuracy of the model.
        # HINT: you can use tf.nn.log_softmax()

        y = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_way), tf.float32)

        q_expand = tf.expand_dims(q_embeddings, axis=1)
        # print("q: ", q_embeddings.shape, "-->", q_expand.shape)

        p_expand = tf.expand_dims(s_prototypes, axis=0)
        # print("p: ", s_prototypes.shape, "-->", p_expand.shape)

        distances = tf.math.pow(
            tf.math.subtract(q_expand, p_expand), 2, name="distances"
        )

        distances = tf.math.reduce_sum(distances, axis=2)
        # (25, 5)

        log_softmax = tf.nn.log_softmax(-distances, axis=1, name="log_softmax")
        # (25, 5)

        log_softmax = tf.reshape(log_softmax, [n_way, n_query, -1])
        # (5, 5, 5)

        loss = -tf.reduce_mean(
            tf.reshape(
                tf.math.reduce_sum(tf.multiply(log_softmax, y_onehot), axis=-1), [-1]
            )
        )

        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(log_softmax, axis=-1), y), dtype=tf.float32)
        )

        ##################################################

        return loss, acc
