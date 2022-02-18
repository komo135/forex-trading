import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Dropout, Concatenate, Lambda
from tensorflow.keras import layers
from network import network

dueling = False


class Output(layers.Layer):
    def __init__(self, output_size):
        super(Output, self).__init__()
        self.output_size = output_size
        self.out = [
            [network.noise(0.1), Dense(32 * output_size), network.layers.Reshape((output_size, 32, 1))]
            for _ in range(output_size)
        ]

    def call(self, inputs, *args, **kwargs):
        out = []
        for out_l in self.out:
            q = out_l[0](inputs)
            q = out_l[1](q)
            q = out_l[2](q)
            out.append(q)
        return tf.concat(out, axis=-1)

    def get_config(self):
        return {"output_size": self.output_size}


network.Output = Output
