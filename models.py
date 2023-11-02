import tensorflow as tf
from tensorflow import keras
from keras import layers
import config as cfg
import random
tf.keras.utils.set_random_seed(cfg.seed)
random.seed(cfg.seed)


def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        encoded_feature = tf.expand_dims(inputs[feature_name], -1)

        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features


def encode_inputs_variable(inputs, encoding_size):
    encoded_features = []
    for feature_name in inputs:
        # Project the numeric feature to encoding_size using linear transformation.
        encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features


def create_model_inputs(feature_names):
    inputs = {}
    for feature_name in feature_names:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )
    return inputs


def create_baseline_model(feature_names):
    inputs = create_model_inputs(feature_names)
    features = encode_inputs(inputs)

    for units in cfg.hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(cfg.dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_wide_and_deep_model(feature_names):
    inputs = create_model_inputs(feature_names)
    wide = encode_inputs(inputs)
    wide = layers.BatchNormalization()(wide)

    deep = encode_inputs(inputs)
    for units in cfg.hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(cfg.dropout_rate)(deep)

    merged = layers.concatenate([wide, deep])
    outputs = layers.Dense(units=1)(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_deep_and_cross_model(feature_names):
    inputs = create_model_inputs(feature_names)
    x0 = encode_inputs(inputs)

    cross = x0
    for _ in cfg.hidden_units:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)

    deep = x0
    for units in cfg.hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(cfg.dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])
    outputs = layers.Dense(units=1)(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super().__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax", name='softmax')

    def call(self, inputs):

        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        softmax_output = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, inp in enumerate(inputs):
            x.append(self.grns[idx](inp))
        grn_output = tf.stack(x, axis=1)

        return softmax_output, grn_output


def variable_selection_model(encoding_size, feature_names):
    inputs = create_model_inputs(feature_names)
    feature_list = encode_inputs_variable(inputs, encoding_size)
    num_features = len(feature_list)

    softmax_output, grn_output = VariableSelection(num_features, encoding_size, cfg.dropout_rate)(
        feature_list
    )
    features = tf.squeeze(tf.matmul(softmax_output, grn_output, transpose_a=True), axis=1)
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
