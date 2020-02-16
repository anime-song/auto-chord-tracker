import keras
import keras.backend as K
import tensorflow as tf
from keras_contrib.layers import CRF
from keras import layers as L
from keras.utils import get_custom_objects


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({'mish': mish})


def _add_lstm_layer(
        n_layers,
        hidden_size,
        dropout=0.4,
        bidirectional=False):

    def wrap(input_layer):
        layer = input_layer

        for i in range(n_layers):
            lstm = L.LSTM(
                hidden_size,
                return_sequences=True,
                dropout=dropout
            )

            if bidirectional:
                layer = L.Bidirectional(lstm)(layer)
            else:
                layer = lstm(layer)

        return layer
    return wrap


def _dense(
        input_layer,
        hidden_size=128):
    return L.Dense(
        hidden_size,
        use_bias=False)(input_layer)


def _dense_bn_relu(
        input_layer,
        hidden_size=128,
        dropout=0.,
        relu_max_value=6.0,
        output_name=None):
    x = _dense(
        input_layer,
        hidden_size=hidden_size)
    x = L.BatchNormalization()(x)
    if relu_max_value != 0.0:
        x = L.ReLU(relu_max_value)(x)

    if output_name is not None:
        x = L.Dropout(dropout, name=output_name)(x)
    else:
        x = L.Dropout(dropout)(x)

    return x


def _dense_bn_mish(input_layer,
                   hidden_size=128,
                   dropout=0.,
                   output_name=None):
    x = _dense(
        input_layer,
        hidden_size=hidden_size)
    x = L.BatchNormalization()(x)
    x = L.Activation(mish)(x)

    if output_name is not None:
        x = L.Dropout(dropout, name=output_name)(x)
    else:
        x = L.Dropout(dropout)(x)

    return x


def chroma_net(
        input_layer,
        hidden_size=128,
        n_layers=1,
        unidirectional=False,
        dropout=0.4):
    
    if unidirectional:
        lstm_hidden_size = hidden_size
    else:
        lstm_hidden_size = hidden_size // 2

    x = input_layer

    x = _dense_bn_mish(x, hidden_size=hidden_size)

    lstm = _add_lstm_layer(
        n_layers=n_layers,
        hidden_size=lstm_hidden_size,
        dropout=dropout,
        bidirectional=not unidirectional)(x)
    x = L.Concatenate()([x, lstm])

    x = _dense_bn_mish(x, hidden_size=hidden_size)

    return x


def smooth_model(layer, output_size=25, output_name=None):
    x = chroma_net(layer, hidden_size=256, n_layers=1, dropout=0.1)
    x = _dense_bn_relu(x, hidden_size=output_size, relu_max_value=1.0, output_name=output_name)

    return x


def ChordNet(
        input_shape=(None, 216),
        hidden_size=256,
        n_layers=1,
        outsize=25,
):
    
    input_layer = L.Input(shape=input_shape)

    x = input_layer
    left = L.Lambda(lambda x: x[:, 0], name="left_spect")(x)
    right = L.Lambda(lambda x: x[:, 1], name="right_spect")(x)

    center = L.Subtract(name="center_spect")([left, right])
    x = L.Concatenate()([left, right, center])

    chroma = chroma_net(
        x,
        hidden_size=hidden_size,
        n_layers=n_layers)

    chord = _dense_bn_relu(chroma, hidden_size=outsize, relu_max_value=1.0, output_name="chord")

    smooth = smooth_model(chord, output_size=outsize, output_name="smooth")

    return keras.Model(inputs=[input_layer], outputs=[chord, smooth])


def crf_model(model, output_size=433):
    bass = CRF(13, learn_mode="marginal")(model.output[1])
    chord = L.Concatenate()([model.output[1], bass])
    chord = CRF(output_size, sparse_target=True)(chord)

    return keras.Model(inputs=[model.input], outputs=[chord, bass])
