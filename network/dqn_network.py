import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from network import network


dueling = False


class Output(layers.Layer):
    def __init__(self, output_size):
        super(Output, self).__init__()
        self.output_size = output_size
        self.out = [
            [Dropout(0.1), layers.Dense(output_size), layers.Reshape((output_size, 1))]
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


# def output(x, output_size):
#     if not dueling:
#         out = []
#         for _ in range(output_size):
#             q = network.noise(network.noise_ratio[-1])(x)
#             q = tf.keras.layers.Dense(output_size, kernel_regularizer=network.norm)(q)
#             q = tf.reshape(q, (-1, output_size, 1))
#             out.append(q)
#         out = tf.keras.layers.Concatenate(axis=-1)(out)
#     else:
#         v_list, a_list = [], []
#         for _ in range(output_size):
#             v = network.noise(network.noise_ratio[-1])(x)
#             v = layers.Dense(1, kernel_regularizer=network.norm)(v)
#             v = tf.reshape(v, (-1, 1, 1))
#
#             a = network.noise(network.noise_ratio[-1])(x)
#             a = layers.Dense(output_size, kernel_regularizer=network.norm)(a)
#             a = tf.reshape(a, (-1, output_size, 1))
#             v_list.append(v)
#             a_list.append(a)
#         a = layers.Concatenate(axis=-1)(a_list)
#         v = layers.Concatenate(axis=1)(v_list)
#
#         a = a - tf.reduce_mean(a, axis=-1, keepdims=True)
#         out = v + a
#
#     return out


network.Output = Output
