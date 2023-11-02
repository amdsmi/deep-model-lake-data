import pandas as pd
import tensorflow as tf
from keras import backend as K
import plotly.express as px


def percentage_error(y_true, y_pred):
    return 100 * tf.reduce_mean(tf.math.abs(y_true - y_pred)) / tf.reduce_mean(tf.math.abs(y_true))


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def mean_absolut_scaled_error(y_true, y_pred):
    diffusion = K.sum(K.abs(y_true - y_pred))
    label_mean = K.mean(y_true)
    variance = K.sum(K.abs(y_true - label_mean))
    return diffusion / variance


def box_plot_monthly(data_path):
    data = pd.read_csv(data_path)
    for key in data.keys()[:-1]:
        fig = px.box(data, x="Date", y=key)
        fig.update_layout(
            title=key,
            xaxis_title="Date",
            yaxis_title="Value",
            font=dict(
                size=18,
                           )
        )
        fig.update_xaxes(tickangle=45)
        fig.write_image(f"plots/{key}_boxplot.jpg")


if __name__ == "__main__":
    box_plot_monthly('data/box_plot_data.csv')
