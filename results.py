from abc import ABC
import pickle
import pandas as pd
from base_class import BaceClass
import tensorflow as tf
import random
import config as cfg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import shap
import matplotlib
import keras
from sklearn.metrics import r2_score
matplotlib.use('TkAgg')
tf.keras.utils.set_random_seed(cfg.seed)
random.seed(cfg.seed)


class ReportResults(BaceClass, ABC):
    def __init__(self,
                 feature_list,
                 target,
                 model_name,
                 normalize):
        super().__init__()
        self.feature_list = feature_list
        self.target = target
        self.model_name = model_name
        self.initialize()
        self.explainer_obj = None
        self.normalize = normalize

    def run_experiment(self):
        if self.normalize:
            self.normalize_data()
        self.declare_model(self.model_name)
        super().run_experiment()
        super().evaluation()

    def plot_history(self):
        pd.DataFrame(self.history.history)[['val_r_squared', 'r_squared']].plot(figsize=(8, 5)).get_figure().savefig(
            f'plots/{self.model_name}_{self.target[0]}_history_r.png')
        pd.DataFrame(self.history.history)[['val_loss', 'loss']].plot(figsize=(8, 5)).get_figure().savefig(
            f'plots/{self.model_name}_{self.target[0]}_history_mse.png')

    def plot_result(self):
        if self.predict is None:
            raise Exception('predicts should be calculated')

        x = self.predict
        y = self.test_data[self.target].values.reshape(-1)
        fig = plt.figure(figsize=(8, 8))
        gs = gd.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        ax_xDist = plt.subplot(gs[0, :2], sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2], sharey=ax_main)
        ax_main.scatter(x, y, marker='.')
        ax_main.set(xlabel="DO_predicted", ylabel="DO_observed")
        ax_xDist.hist(x, bins=100, align='mid')
        ax_xDist.set(ylabel='Do_predicted')
        ax_xCumDist = ax_xDist.twinx()
        ax_xCumDist.hist(x, bins=100, cumulative=True, histtype='step', density=True, color='r', align='mid')
        ax_xCumDist.tick_params('y', colors='r')
        ax_xCumDist.set_ylabel('cumulative', color='r')

        ax_yDist.hist(y, bins=100, orientation='horizontal', align='mid')
        ax_yDist.set(xlabel='Do_observed')
        ax_yCumDist = ax_yDist.twiny()
        ax_yCumDist.hist(y, bins=100, cumulative=True, histtype='step', density=True, color='r', align='mid',
                         orientation='horizontal')
        ax_yCumDist.tick_params('x', colors='r')
        ax_yCumDist.set_xlabel('cumulative', color='r')

        fig.savefig(f'plots/{self.model_name}_{self.target[0]}_pred_vs_real.png')

    def make_explainer(self, sample_num):

        explainer_ = shap.KernelExplainer(model=self.shap_prediction,
                                          data=shap.sample(
                                              self.test_data[cfg.feature_list][:sample_num],
                                              cfg.shap_sample_num
                                          ),
                                          link="identity")
        self.explainer_obj = explainer_(base_instance.test_data[cfg.feature_list][:sample_num])

    def force_plot(self, sample_num):
        explainer_ = shap.KernelExplainer(model=self.shap_prediction,
                                          data=self.test_data[cfg.feature_list][:sample_num], link="identity")
        shap_value = explainer_.shap_values(base_instance.test_data[cfg.feature_list][:sample_num])
        shap.initjs()
        shap_plot = shap.plots.force(explainer_.expected_value, shap_values=shap_value, feature_names=self.feature_list)
        shap.save_html("plots/chl_a_index.htm", shap_plot)

    def dependence_plot(self, sample_num):
        explainer_ = shap.KernelExplainer(model=self.shap_prediction,
                                          data=shap.sample(
                                              self.test_data[cfg.feature_list][:sample_num],
                                              cfg.shap_sample_num
                                          ),
                                          link="identity")
        shap_value = explainer_.shap_values(base_instance.test_data[cfg.feature_list][:sample_num])
        shap.dependence_plot(ind=4, shap_values=shap_value, features=self.test_data[cfg.feature_list][:sample_num],
                             feature_names=self.feature_list)
        # shap.save_html("plots/chl_a_index.htm", shap_plot)

    def bar_plot(self):
        return shap.plots.bar(shap_values=self.explainer_obj)

    def waterfall_plot(self):
        return shap.plots.waterfall(shap_values=self.explainer_obj[0])

    def violin_plot(self):
        return shap.plots.violin(shap_values=self.explainer_obj)

    def beeswarm_plot(self):
        return shap.plots.beeswarm(shap_values=self.explainer_obj)

    def heatmap_plot(self):
        return shap.plots.heatmap(shap_values=self.explainer_obj)

    def softmax_output(self):
        if self.model_name in ['base_line', 'wide_deep', 'deep_cross']:
            raise Exception('only variable selection model is supported')
        softmax_result = pd.DataFrame()
        inp = self.model.input
        out = self.model.get_layer('variable_selection').output
        new_model = keras.Model(inputs=inp, outputs=out)
        for layer in new_model.layers:
            print(layer.name)
        softmax_output, grn_output = new_model.predict([self.test_data[[feature]] for feature in self.feature_list])

        for i in range(len(self.feature_list)):
            softmax_result[self.feature_list[i]] = softmax_output[:, i].reshape(-1)

        return softmax_result

    def save_explainer(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.explainer_obj, file)
            print(f'Object successfully saved to "{file_name}"')

    def load_explainer(self, file_path):
        with open(file_path, 'rb') as file:
            self.explainer_obj = pickle.load(file)

    def print_results(self):
        # (self.train_data - self._min_data) / (self._max_data - self._min_data)
        predict_test = self.test_prediction()
        denormal_pred = (predict_test * (self.target_max - self.target_min)) + self.target_min
        denormal_true = (self.test_data[self.target].values * (self.target_max - self.target_min)) + self.target_min

        return r2_score(denormal_true.reshape(-1), denormal_pred.reshape(-1))


if __name__ == "__main__":
    base_instance = ReportResults(
        feature_list=cfg.feature_list,
        target=cfg.target,
        model_name=cfg.model_name,
        normalize=cfg.normalize
    )
    base_instance.run_experiment()

    # base_instance.make_explainer(-1)
    # base_instance.bar_plot()
    # base_instance.violin_plot()
    # base_instance.beeswarm_plot()
    # base_instance.save_explainer('results/do_explainer_all_features')

    # base_instance.correlation_plot()
    # all_prediction = base_instance.all_data_prediction()
    # base_instance._data[f'{base_instance.target[0]}_{base_instance.model_name}'] = all_prediction
    # base_instance._data.to_csv(f'results/{base_instance.target[0]}_{base_instance.model_name}.csv')
    # base_instance.dependence_plot(1000)
    # base_instance.make_explainer(-1)
    # base_instance.bar_plot()
    # base_instance.violin_plot()
    # base_instance.beeswarm_plot()
    # base_instance.save_explainer('results/do_explainer')
    # base_instance.heatmap_plot()
    # print(sns.barplot(base_instance.softmax_output(), orient='h', color='red').get_figure().savefig('chl_a.png'))
