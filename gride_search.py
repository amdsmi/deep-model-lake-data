from abc import ABC
from base_class import BaceClass
import random
import config as cfg
import tensorflow as tf
import pandas as pd
import os
from itertools import chain, combinations

tf.keras.utils.set_random_seed(cfg.seed)
random.seed(cfg.seed)


class GridSearch(BaceClass, ABC):
    def __init__(self, save_result_path):
        super().__init__()
        self.save_result_path = save_result_path
        self._grid_search_space = None
        self.grid_search_report = {}
        self.initialize()
        self.feature_selector()

    def feature_selector(self):
        if self._grid_search_space is not None:
            raise Exception('the features are already selected')
        feature_target = {}
        if cfg.simple_grid:
            for i in cfg.TARGET_FEATURE_NAMEs:
                feature_target[i] = chain.from_iterable(
                    combinations(cfg.feature_list, r) for r in range(1, len(cfg.feature_list) + 1)
                )

        else:
            for i in cfg.TARGET_FEATURE_NAMEs:
                temp_list = []
                for j in range(len(cfg.ONE_OF_FEATURE_NAMEs) + 1):
                    for k in range(len(cfg.VARIABLE_FEATURE_NAMEs)):
                        for l in combinations(cfg.VARIABLE_FEATURE_NAMEs, k + 1):
                            feature = [*cfg.STEADY_FEATURE_NAMEs, *l]
                            try:
                                feature.append(cfg.ONE_OF_FEATURE_NAMEs[j])
                                temp_list.append(feature)
                            except:
                                temp_list.append(feature)

                feature_target[i] = temp_list

        self._grid_search_space = feature_target

    def report_to_csv(self):
        with pd.ExcelWriter(os.path.join(self.save_result_path)) as writer:
            for key, value in self.grid_search_report.items():
                pd.DataFrame.from_dict(value).T.to_excel(writer, sheet_name=key)

    def run_experiment(self):
        if self._grid_search_space is None:
            raise Exception('there is no grid data')
        for target, grid in self._grid_search_space.items():
            self.target = target
            sub_results = {}
            for feature in grid:
                sub_results['_'.join(feature)] = {}
                for model_name in cfg.MODEL_LIST:
                    self.feature_list = feature
                    self.declare_model(model_name)
                    super().run_experiment()
                    super().evaluation()

                    sub_results['_'.join(feature)].update(
                        {model_name + key: value for key, value in self.metrics.items()}
                    )

            self.grid_search_report[target] = sub_results


if __name__ == '__main__':
    grid_search = GridSearch(save_result_path=cfg.grid_search_result_path)
    grid_search.run_experiment()
    grid_search.report_to_csv()
