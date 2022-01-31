import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Layer, Concatenate, MultiHeadAttention, LayerNormalization, ELU, Add, Dense
from tensorflow.keras import layers, Sequential
from DepthWiseConv1D import DepthwiseConv1D
import os

normal_model = keras.Model

norm = tf.keras.regularizers.l1_l2(l1=0, l2=0)
noise_ratio = (0.1, 0.2, 0.2, 0.1)
noise = layers.GaussianDropout

gamma = [1, 1.2, 1.4, 1.7, 2., 2.4, 2.9, 3.5]
alpha = [1, 1.1, 1.21, 1.33, 1.46, 1.61, 1.77, 1.94]

l_b0 = (3, 4, 6, 3)
noise_b0 = [0.1, 0.1, 0.1, 0.1]
noise_b1 = [0.1, 0.114, 0.114, 0.1]
noise_b2 = [0.1, 0.128, 0.128, 0.1]
noise_b3 = [0.1, 0.142, 0.142, 0.1]
noise_b4 = [0.1, 0.157, 0.157, 0.1]
noise_b5 = [0.1, 0.171, 0.171, 0.1]
noise_b6 = [0.1, 0.185, 0.185, 0.1]
noise_b7 = [0.1, 0.2, 0.2, 0.1]
noise_l = [noise_b0, noise_b1, noise_b2, noise_b3, noise_b4, noise_b5, noise_b6, noise_b7]


def layer(layer_name: str, dim, use_bias=True, kernel_size=1, groups=1):
    layer_name = layer_name.lower()
    if layer_name == "conv1d":
        return Conv1D(dim, kernel_size, 1, "same", kernel_initializer="he_normal", use_bias=use_bias, groups=groups)
    elif layer_name == "lambdalayer":
        return LambdaLayer(dim)
    elif layer_name == "skconv":
        return SKConv(dim, groups)
    elif layer_name == "depthwiseconv1d":
        return DepthwiseConv1D(kernel_size, 1, "same", kernel_initializer="he_normal", use_bias=use_bias)


class Mish(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return inputs * tf.tanh(tf.math.softplus(inputs))


class Activation(layers.Layer):
    def __init__(self, activation="swish"):
        super(Activation, self).__init__()
        self.activation = activation.lower()
        if self.activation == "mish":
            self.act = Mish()
        else:
            self.act = layers.Activation(activation)
        self.normalization = LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.normalization(inputs)
        return self.act(x)

    def get_config(self):
        return {"activation": self.activation}


def inputs_f(input_shape, dim, kernel_size, strides, pooling, padding="same"):
    inputs = tf.keras.layers.Input(input_shape)
    x = noise(noise_ratio[0])(inputs)
    # x = tf.keras.layers.GaussianNoise(0.05)(inputs)
    x = layers.Conv1D(dim, kernel_size, strides, padding, kernel_initializer="he_normal", kernel_regularizer=norm)(x)
    if pooling:
        x = tf.keras.layers.AvgPool1D(3, 2)(x)

    return inputs, x


def output(x, output_size):
    x = tf.keras.layers.Dense(output_size)(x)
    return x


class Pyconv(Layer):
    def __init__(self, dim, groups=32):
        super(Pyconv, self).__init__()

        self.dim = dim
        self.groups = groups

        self.k = k = [3, 5, 7, 9]
        self.conv = [
            Conv1D(dim // 4, k, 1, "same", kernel_initializer="he_normal", groups=groups, kernel_regularizer=norm) for k
            in k
        ]
        self.concat = Concatenate()

    def call(self, inputs, *args, **kwargs):
        x = []
        for conv in self.conv:
            x.append(conv(inputs))

        return self.concat(x)

    def get_config(self):
        new_config = {
            "dim": self.dim,
            "groups": self.groups
        }
        config = {}
        # config = super(Pyconv, self).get_config()
        config.update(new_config)

        return config


class SE(Layer):
    def __init__(self, dim, r=.25):
        super(SE, self).__init__()
        self.dim = dim
        self.r = r

        self.mlp = tf.keras.Sequential([
            Dense(int(dim * r), "relu", kernel_initializer="he_normal", kernel_regularizer=norm),
            Dense(dim, "sigmoid")
        ])

    def call(self, inputs, *args, **kwargs):
        x = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x = self.mlp(x)
        x *= inputs

        return x

    def get_config(self):
        config = {
            "dim": self.dim,
            "r": self.r
        }

        return config


class CBAM(Layer):
    def __init__(self, filters, r=.25):
        super(CBAM, self).__init__()
        self.filters = filters
        self.r = r

        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.mlp = [
            tf.keras.layers.Dense(int(filters * r), "relu", kernel_initializer="he_normal", kernel_regularizer=norm),
            tf.keras.layers.Dense(filters, kernel_regularizer=norm)
        ]

        self.concat = tf.keras.layers.Concatenate()
        self.conv = tf.keras.layers.Conv1D(1, 7, 1, "same", activation="sigmoid", kernel_regularizer=norm)

    def compute_mlp(self, x, pool):
        x = pool(x)
        for mlp in self.mlp:
            x = mlp(x)

        return x

    def call(self, inputs, training=None, *args, **kwargs):
        x = self.compute_mlp(inputs, self.avg_pool) + self.compute_mlp(inputs, self.max_pool)
        x = inputs * tf.reshape(tf.nn.sigmoid(x), (-1, 1, self.filters))

        conv = self.concat([tf.reduce_mean(x, -1, keepdims=True), tf.reduce_max(x, -1, keepdims=True)])
        return x * self.conv(conv)

    def get_config(self):
        new_config = {"filters": self.filters,
                      "r": self.r}
        config = {}
        # config = super(CBAM, self).get_config()
        config.update(new_config)
        return config


class PositionAdd(Layer):
    def build(self, input_shape):
        self.pe = self.add_weight("pe", [input_shape[1], input_shape[2]],
                                  initializer=tf.keras.initializers.zeros())

    def call(self, inputs, **kwargs):
        return inputs + self.pe


class Attention(Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 dropout=noise_ratio[1],
                 use_bias=True,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._dim = dim
        self._dropout = dropout
        self._use_bias = use_bias

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=use_bias)
        self.qkv_reshape = layers.Reshape((-1, num_heads, head_dim))
        self.qv_permute = layers.Permute((2, 1, 3))
        self.k_permute = layers.Permute((2, 3, 1))
        self.attn_reshape = keras.Sequential([
            layers.Permute((2, 1, 3)),
            layers.Reshape((-1, dim))
        ])
        self.proj = layers.Dense(dim)
        self.drop_out = noise(dropout)

    def call(self, inputs: tf.Tensor, training=False, *args, **kwargs):
        qkv = self.qkv(inputs)
        q, k, v = tf.split(qkv, 3, -1)

        q = self.qkv_reshape(q)
        k = self.qkv_reshape(k)
        v = self.qkv_reshape(v)

        q = self.qv_permute(q)
        k = self.k_permute(k)
        v = self.qv_permute(v)

        attn = tf.matmul(q, k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        x = tf.matmul(attn, v)
        x = self.attn_reshape(x)
        x = self.proj(x)
        x = self.drop_out(x)

        return x

    def get_config(self):
        config = {}
        # config = super(Attention, self).get_config()
        new_config = {
            "num_heads": self._num_heads,
            "dim": self._dim,
            "noise_ratio": self._dropout,
            "use_bias": self._use_bias
        }
        config.update(new_config)
        return config


class TransformerMlp(Layer):
    def __init__(self, dim, mlp_dim):
        super(TransformerMlp, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.dense = Sequential([
            Dense(mlp_dim, "gelu"),
            noise(noise_ratio[2]),
            Dense(dim),
            noise(noise_ratio[2])
        ])

    def call(self, inputs, *args, **kwargs):
        return self.dense(inputs)

    def get_config(self):
        config = {}
        config.update({
            "dim": self.dim,
            "mlp_dim": self.mlp_dim
        })

        return config


class Transformer(Layer):
    def __init__(self, dim, mlp_dim, heads, use_bias=False):
        super(Transformer, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.heads = heads
        self.use_bias = use_bias

        self.attn = Sequential([LayerNormalization(), Attention(dim, heads, noise_ratio[1], use_bias)])
        self.mlp = Sequential([LayerNormalization(), TransformerMlp(dim, mlp_dim)])

    def call(self, inputs, *args, **kwargs):
        x = self.attn(inputs) + inputs
        x = self.mlp(x) + x

        return x

    def get_config(self):
        config = {
            "dim": self.dim,
            "mlp_dim": self.mlp_dim,
            "heads": self.heads,
            "use_bias": self.use_bias
        }

        return config


class LambdaLayer(Layer):

    def __init__(self, out_dim, heads=4, use_bias=False, u=4, kernel_size=7):
        super(LambdaLayer, self).__init__()

        self.out_dim = out_dim
        k = 16
        self.heads = heads
        self.v = out_dim // heads
        self.u = u
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.top_q = tf.keras.layers.Conv1D(k * heads, 1, 1, "same", use_bias=use_bias)
        self.top_k = tf.keras.layers.Conv1D(k * u, 1, 1, "same", use_bias=use_bias)
        self.top_v = tf.keras.layers.Conv1D(self.v * self.u, 1, 1, "same", use_bias=use_bias)

        self.norm_q = tf.keras.layers.LayerNormalization()
        self.norm_v = tf.keras.layers.LayerNormalization()

        self.rearrange_q = keras.Sequential([
            layers.Reshape((-1, heads, k)),
            layers.Permute((2, 3, 1))
        ])
        self.rearrange_k = keras.Sequential([
            layers.Reshape((-1, u, k)),
            layers.Permute((2, 3, 1))
        ])
        self.rearrange_v = keras.Sequential([
            layers.Reshape((-1, u, self.v)),
            layers.Permute((2, 3, 1))
        ])

        self.rearrange_v2 = layers.Permute((2, 3, 1))
        self.rearrange_lp = layers.Permute((1, 3, 2))
        self.rearrange_output = layers.Reshape((-1, out_dim))

        # self.rearrange_q = Rearrange("b n (h k) -> b h k n", h=self.heads)
        # self.rearrange_k = Rearrange("b n (u k) -> b u k n", u=u)
        # self.rearrange_v = Rearrange("b n (u v) -> b u v n", u=u)
        #
        # self.rearrange_v2 = Rearrange("b u v n -> b v n u")
        # self.rearrange_lp = Rearrange("b v n k -> b v k n")
        # self.rearrange_output = Rearrange("b n h v -> b n (h v)")

        self.pos_conv = tf.keras.layers.Conv2D(k, (1, self.kernel_size), padding="same")

    def call(self, inputs, *args, **kwargs):
        q = self.top_q(inputs)
        k = self.top_k(inputs)
        v = self.top_v(inputs)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = self.rearrange_q(q)
        k = self.rearrange_k(k)
        v = self.rearrange_v(v)

        k = tf.nn.softmax(k)

        lc = tf.einsum("b u k n, b u v n -> b k v", k, v)
        yc = tf.einsum("b h k n, b k v -> b n h v", q, lc)

        v = self.rearrange_v2(v)
        lp = self.pos_conv(v)
        lp = self.rearrange_lp(lp)
        yp = tf.einsum("b h k n, b v k n -> b n h v", q, lp)

        y = yc + yp
        output = self.rearrange_output(y)

        return output

    # def compute_output_shape(self, input_shape):
    #     return input_shape[0], self.dim

    def get_config(self):
        new_config = {
            "out_dim": self.out_dim,
            "heads": self.heads,
            "use_bias": self.use_bias,
            "u": self.u,
            "kernel_size": self.kernel_size
        }
        config = {}
        # config = super(LambdaLayer, self).get_config()
        config.update(new_config)

        return config


class SKConv(Layer):
    def __init__(self, filters: int, groups=32, r=16):
        super(SKConv, self).__init__()
        self.filters = filters
        self.z_filters = np.maximum(filters // r, 32)
        self.groups = groups
        self.r = r

        self.u1 = layers.Conv1D(filters, 3, 1, "same", kernel_initializer="he_normal", groups=groups)
        self.u2 = layers.Conv1D(filters, 5, 1, "same", kernel_initializer="he_normal", dilation_rate=1, groups=groups)

        self.add = layers.Add(axis=-1)

        self.z = layers.Dense(self.z_filters, "elu", kernel_initializer="he_normal")

        self.a = layers.Dense(self.filters)
        self.b = layers.Dense(self.filters)
        self.concat = layers.Concatenate(axis=1)

    def call(self, inputs, *args, **kwargs):
        u1 = self.u1(inputs)
        u2 = self.u2(inputs)

        u = self.add([u1, u2])
        s = tf.reduce_mean(u, axis=1, keepdims=True)
        z = self.z(s)
        a = self.a(z)
        b = self.b(z)
        ab = self.concat([a, b])
        ab = tf.nn.softmax(ab, axis=1)
        a, b = tf.split(ab, 2, 1)

        u1 *= a
        u2 *= b

        return u1 + u2

    def get_config(self):
        config = {
            "filters": self.filters,
            "groups": self.groups,
            "r": self.r
        }

        return config


class MBBlock(layers.Layer):
    def __init__(self, idim, odim, expand_ratio, kernel_size, se_ratio=0.25, layer_name="DepthwiseConv1D",
                 types="resnet"):
        super(MBBlock, self).__init__()
        self.idim = idim
        self.odim = odim
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.kernel_size = kernel_size
        self.layer_name = layer_name
        self.types = types.lower()
        assert self.types == "resnet" or self.types == "densenet"

        self.l1 = keras.Sequential([
            Activation(),
            noise(noise_ratio[1]),
            Conv1D(int(idim * expand_ratio), 1, 1, "same", use_bias=False, kernel_initializer="he_normal"),
            Activation(),
        ])
        self.l2 = layer(layer_name, int(idim * expand_ratio), False, kernel_size, 1)
        self.l3 = keras.Sequential([
            Activation(),
            SE(int(idim * expand_ratio), .25),
            noise(noise_ratio[2]),
            Conv1D(odim, 1, 1, "same", use_bias=False, kernel_initializer="he_normal"),
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)

        if self.types == "resnet":
            return x + inputs
        elif self.types == "densenet":
            return tf.concat([inputs, x], axis=-1)

    def get_config(self):
        config = {
            "idim": self.idim,
            "odim": self.odim,
            "expand_ratio": self.expand_ratio,
            "se_ratio": self.se_ratio,
            "kernel_size": self.kernel_size,
            "layer_name": self.layer_name,
            "types": self.types
        }
        return config


class FuseBlock(Layer):
    def __init__(self, idim, odim, expand_ratio, kernel_size, se_ratio=0, *args, **kwargs):
        super(FuseBlock, self).__init__()
        self.idim = idim
        self.odim = odim
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.se_ratio = se_ratio

        self.l1 = keras.Sequential([
            LayerNormalization(),
            layers.Activation("swish"),
            noise(noise_ratio[1]),
            Conv1D(int(idim * expand_ratio), kernel_size, 1, "same", kernel_initializer="he_normal", use_bias=False)
        ])
        if self.se_ratio != 0:
            self.se = SE(int(idim * expand_ratio), se_ratio)
        self.l2 = keras.Sequential([
            LayerNormalization(),
            layers.Activation("swish"),
            noise(noise_ratio[2]),
            Conv1D(odim, 1, 1, "same", kernel_initializer="he_normal", use_bias=False)
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.l1(inputs)
        if self.se_ratio != 0:
            x = self.se(x)
        x = self.l2(x)

        return x + inputs

    def get_config(self):
        config = {
            "idim": self.idim,
            "odim": self.odim,
            "expand_ratio": self.expand_ratio,
            "se_ratio": self.se_ratio,
            "kernel_size": self.kernel_size
        }
        return config


class ConvBlock(layers.Layer):
    def __init__(self, dim, layer_name="Conv1D", types="", groups=1, bias=True, se=False, cbam=False):
        """
        :param dim: output dimention
        :param layer_name: layer name
        :param types: "densenet" or "resnet"
        """
        super(ConvBlock, self).__init__()

        self.dim = dim
        self.layer_name = layer_name
        self.types = types.lower()
        self.groups = groups
        self.bias = bias
        self.se = se
        self.cbam = cbam

        assert self.types == "densenet" or self.types == "resnet"

        conv1 = layer("conv1d", dim * 4, bias, 3, groups) if types == "densenet" else layer("conv1d", dim, True, 1, 1)

        self.layer = [
            Activation(),
            noise(noise_ratio[1]),
            conv1,
            Activation(),
            noise(noise_ratio[2]),
            layer(layer_name, dim // (2 if self.types == "resnet" else 1), bias, 3, groups)
        ]

        if self.types == "resnet":
            self.layer.extend([
                Activation(),
                noise(noise_ratio[1]),
                layer("conv1d", dim, True, 1, 1),
            ])

        if self.se:
            self.se_net = SE(dim)
        elif self.cbam:
            self.cbam_net = CBAM(dim)

    def call(self, inputs, *args, **kwargs):
        x = inputs

        for l in self.layer:
            x = l(x)

        if self.se:
            x = self.se_net(x)
        elif self.cbam:
            x = self.cbam_net(x)

        if self.types == "densenet":
            return tf.concat([inputs, x], axis=-1)
        elif self.types == "resnet":
            return tf.add(inputs, x)

    def get_config(self):
        config = {
            "dim": self.dim,
            "layer_name": self.layer_name,
            "types": self.types,
            "groups": self.groups,
            "bias": self.bias,
            "se": self.se,
            "cbam": self.cbam
        }
        return config


class ConvnextBlock(layers.Layer):
    def __init__(self, dim: int, layer_name: str, types: str, se=False, cbam=False):
        super(ConvnextBlock, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        self.types = types
        self.se = se
        self.cbam = cbam

        self.l = [
            layer(layer_name, dim, True, 7),
            layers.LayerNormalization(),
            layers.Conv1D(dim * 4, 1, 1, "same", kernel_initializer="he_normal"),
            layers.Activation("gelu"),
            layers.Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")
        ]
        if self.se:
            self.se_layer = SE(dim)
        elif self.cbam:
            self.cbam_layer = CBAM(dim)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for l in self.l:
            x = l(x)

        if self.se:
            self.se_layer(x)
        elif self.cbam:
            self.cbam_layer(x)

        if self.types == "resnet":
            return inputs + x
        elif self.types == "densenet":
            return tf.concat([inputs, x], -1)

    def get_config(self):
        config = {
            "dim": self.dim,
            "layer_name": self.layer_name,
            "types": self.types,
            "se": self.se,
            "cbam": self.cbam
        }
        return config


class SAMModel(keras.Model):
    rho = 0.05
    def train_step(self, data):
        x, y = data
        e_ws = []

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        if "_optimizer" in dir(self.optimizer):  # mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables,
                                                                          tape=tape)
        else:
            grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)

        grads = [g for g, _ in grads_and_vars]

        grad_norm = self._grad_norm(grads)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(grads, self.trainable_variables):
            e_w = grad * scale
            e_ws.append(e_w)
            param.assign_add(e_w)

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        if "_optimizer" in dir(self.optimizer):  # mixed float policy
            grads_and_vars = self.optimizer._optimizer._compute_gradients(loss, var_list=self.trainable_variables,
                                                                          tape=tape)
        else:
            grads_and_vars = self.optimizer._compute_gradients(loss, var_list=self.trainable_variables, tape=tape)
        grads = [g for g, _ in grads_and_vars]

        for e_w, param in zip(e_ws, self.trainable_variables):
            param.assign_sub(e_w)

        grads_and_vars = list(zip(grads, self.trainable_variables))

        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(y, predictions)

        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm

    def call(self, inputs, training=None, mask=None):
        return super(SAMModel, self).call(inputs, training, mask)

    def get_config(self):
        return super(SAMModel, self).get_config()


class Model:
    def __init__(self, num_layer, dim: int, layer_name: str, types: str, scale=0,
                 groups=1, sam=False, se=False, cbam=False, efficientv1=False, efficientv2=False, vit=False,
                 convnext=False):
        global noise_ratio
        self.num_layer = num_layer
        self.layer_name = layer_name.lower()
        self.types = types
        self.groups = groups
        self.sam = bool(sam)
        self.se = bool(se)
        self.cbam = bool(cbam)
        self.efficientv1 = bool(efficientv1)
        self.efficientv2 = bool(efficientv2)
        self.vit = vit
        self.convnext = convnext

        noise_ratio = noise_l[scale]
        self.gamma = gamma[scale]
        self.alpha = alpha[scale]
        self.dim = int(dim * self.gamma) if dim else dim
        self.num_layer = num_layer if num_layer else np.round(np.array(l_b0) * self.alpha).astype(int)
        noise_ratio = noise_l[scale]

        if efficientv1 or efficientv2:
            self.dim = int((16 if self.types == "resnet" else 32) * self.gamma)
        if self.dim and self.layer_name == "lambdalayer":
            self.dim = 4 * int(np.round(self.dim / 4))

    def build_eff_block(self, l):
        block = None
        if self.efficientv1:
            block = [MBBlock for _ in range(len(l))]
        elif self.efficientv2:
            block = [FuseBlock, FuseBlock, FuseBlock]
            block.extend([MBBlock for _ in range(len(l) - 3)])

        self.block = block

    def transition(self, x, dim=None, pool=True):
        if self.types == "densenet":
            dim = x.shape[-1] // 2 if pool else x.shape[-1]
        elif self.types == "resnet":
            dim = self.dim = self.dim * 2 if dim is None else dim

        if self.convnext:
            x = layers.LayerNormalization()(x)
            x = layers.Conv1D(dim, 2, 2, "same", kernel_initializer="he_normal")(x)
            x = layers.LayerNormalization()(x)
        else:
            x = Activation()(x)
            x = Conv1D(dim, 1, 1, "same", kernel_initializer="he_normal")(x)
            if pool:
                x = tf.keras.layers.AvgPool1D()(x)

        return x

    def efficient_model(self, x):
        l = [1, 2, 2, 3, 3, 4, 1]
        k = [3, 3, 5, 3, 5, 5, 3]
        pool = [False, False, True, True, True, False, True]
        self.build_eff_block(l)

        if self.types == "resnet":
            ic = [16, 16, 24, 40, 80, 112, 192]
            oc = [16, 24, 40, 80, 112, 192, 320]
            ep = [1, 6, 6, 6, 6, 6, 6]
        else:
            ic = [32 for _ in range(len(l))]
            oc = ic
            ep = [6 for _ in range(len(l))]

        ic = (np.array(ic) * self.gamma).astype(np.int32)
        oc = (np.array(oc) * self.gamma).astype(np.int32)
        l = np.round(np.array(l) * self.alpha).astype(np.int32)

        if self.layer_name == "lambdalayer":
            ic = [int(4 * np.round(ic / 4)) for ic in ic]
            oc = [int(4 * np.round(oc / 4)) for oc in oc]

        for e, (ic, oc, ep, l, k, pool, block) in enumerate(zip(ic, oc, ep, l, k, pool, self.block)):

            if e != 0:
                x = self.transition(x, oc, pool)

            for _ in range(l):
                x = block(ic, oc, ep, k, 0.25, self.layer_name, self.types)(x)

        return x

    def conv_model(self, x):

        for i, l in enumerate(self.num_layer):
            if i != 0:
                x = self.transition(x, None, True)

            for _ in range(l):
                if self.convnext:
                    x = ConvnextBlock(self.dim, self.layer_name, self.types)(x)
                else:
                    x = ConvBlock(self.dim, self.layer_name, self.types, self.groups, True, self.se, self.cbam)(x)

        return x

    def build_model(self, input_shape, output_size):
        inputs, x = inputs_f(input_shape, self.dim, 5, 1, False, "same")

        if self.efficientv1 or self.efficientv2:
            x = self.efficient_model(x)
        else:
            x = self.conv_model(x)

        x = tf.keras.layers.GlobalAvgPool1D()(x)
        x = layers.LayerNormalization()(x)
        x = output(x, output_size)

        return SAMModel(inputs, x) if self.sam else normal_model(inputs, x)
