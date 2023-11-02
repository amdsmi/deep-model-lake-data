import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod
from models import variable_selection_model, create_baseline_model, create_wide_and_deep_model, \
    create_deep_and_cross_model
import config as cfg
from utils import mean_absolut_scaled_error, r_squared
import random
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
tf.keras.utils.set_random_seed(cfg.seed)
random.seed(cfg.seed)


class BaceClass(ABC):
    def __init__(self):
        self.data_path = cfg.data_path
        self.feature_list = None
        self.target = None
        self.model_name = None
        self._save_directory_name = None
        self._is_trained = None
        self.model = None
        self._data = None
        self.train_data = None
        self.test_data = None
        self.valid_data = None
        self.predict = None
        self.metrics = None
        self.history = None
        self._date = None
        self._min_data = None
        self._max_data = None
        self.target_min = None
        self.target_max = None

    def initialize(self):
        self.data_reader()
        self.data_splitter()
        self.min_max()

    def declare_model(self, model_name):
        if model_name == 'base_line':
            self.model = create_baseline_model(self.feature_list)

        elif model_name == 'wide_deep':
            self.model = create_wide_and_deep_model(self.feature_list)

        elif model_name == 'deep_cross':
            self.model = create_deep_and_cross_model(self.feature_list)

        else:
            self.model = variable_selection_model(cfg.encoding_size, self.feature_list)

    def data_reader(self):
        xls = pd.ExcelFile(self.data_path)

        multi_data = []
        for name in xls.sheet_names:
            multi_data.append(pd.read_excel(xls, name))
        data = pd.concat(multi_data)
        data = data.rename(columns=cfg.MAPPING_NAME)
        data = shuffle(data, random_state=cfg.seed)
        turb = data[data['Turb'] < 6]
        data = turb[turb['Chl_a'] < 20]
        self._date = self.date_correct(data['Date.1'])
        data = data[cfg.all_features]
        self._data = data.astype('float32')
        self._data.to_csv('visual_data.csv')

    def min_max(self):
        self._min_data = self.train_data.min()
        self._max_data = self.train_data.max()
        self.target_min = self.train_data[self.target].min().values[0]
        self.target_max = self.train_data[self.target].max().values[0]

    def normalize_data(self):
        self.train_data = (self.train_data-self._min_data)/(self._max_data-self._min_data)
        self.test_data = (self.test_data-self._min_data)/(self._max_data-self._min_data)
        self.valid_data = (self.valid_data-self._min_data)/(self._max_data-self._min_data)

    @staticmethod
    def date_correct(date_column):
        return date_column.apply(lambda x: datetime.datetime(2023, 4, 20) if x == '20 Ap' else x)

    def data_splitter(self):

        data = self._data.copy()
        random_selection = np.random.rand(len(data.index)) <= 0.7
        self.train_data = data[random_selection]
        test_valid_data = data[~random_selection]
        split = len(test_valid_data) // 2
        self.valid_data = test_valid_data[:split]
        self.test_data = test_valid_data[split:]

    def run_experiment(self):
        if self.model is None:
            raise Exception('the model note define yet')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[r_squared, mean_absolut_scaled_error, keras.metrics.RootMeanSquaredError()],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.buffer_epochs, restore_best_weights=True
        )

        self.history = self.model.fit(x=[self.train_data[[feature]] for feature in self.feature_list],
                                      y=self.train_data[self.target],
                                      epochs=cfg.num_epochs,
                                      validation_data=([self.valid_data[[feature]] for feature in self.feature_list],
                                                       self.valid_data[self.target]),
                                      callbacks=[early_stopping])
        self._is_trained = True

    def get_predict(self):
        if self.model is None:
            raise Exception('the model not define yet')
        if self._is_trained is None:
            raise Exception('the model not train yet')

        self.predict = self.model.predict(x=[self.test_data[[feature]] for feature in self.feature_list]).flatten()

    def evaluation(self):
        if self.model is None:
            raise Exception('the model not define yet')
        if self._is_trained is None:
            raise Exception('the model not train yet')

        train_results = self.model.evaluate(x=[self.train_data[[feature]] for feature in self.feature_list],
                                            y=self.train_data[self.target])

        test_results = self.model.evaluate(x=[self.test_data[[feature]] for feature in self.feature_list],
                                           y=self.test_data[self.target])

        valid_results = self.model.evaluate(x=[self.valid_data[[feature]] for feature in self.feature_list],
                                            y=self.valid_data[self.target])

        self.metrics = {'train_MSE': train_results[0],
                        'train_r2': train_results[1],
                        'train_MAE': train_results[2],
                        'train_RMSE': train_results[3],

                        'test_MSE': test_results[0],
                        'test_r2': test_results[1],
                        'test_MAE': test_results[2],
                        'test_RMSE': test_results[3],

                        'validation_MSE': valid_results[0],
                        'validation_r2': valid_results[1],
                        'validation_MAE': valid_results[2],
                        'validation_RMSE': valid_results[3]
                        }

    def save_model(self):
        if self.model is None:
            raise Exception('the model not define yet')
        if self._is_trained is None:
            raise Exception('the model not train yet')
        self.directory_name()
        tf.saved_model.save(self.model, self._save_directory_name)

    def load_model(self):
        self.directory_name()
        self.model = tf.saved_model.load(self._save_directory_name)

    def plot_model(self):
        if self.model is None:
            raise Exception('the model not define yet')
        tf.keras.utils.plot_model(self.model)

    def shap_prediction(self, shap_data):
        return self.model.predict([shap_data[:, i] for i in range(shap_data.shape[1])]).flatten()

    def correlation_plot(self):
        data = self._data.copy()
        data = data.drop(columns=['Lat', 'Long', 'Quant'])
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, annot=True, mask=mask, cmap=cmap).get_figure().savefig('plots/correlation.png')

    def all_data_prediction(self):
        return self.model.predict(x=[self._data[[feature]] for feature in self.feature_list]).flatten()

    def test_prediction(self):
        return self.model.predict(x=[self.test_data[[feature]] for feature in self.feature_list]).flatten()
