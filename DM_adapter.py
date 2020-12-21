from __future__ import print_function

import os

import deepmatcher as dm
import nltk
import pandas as pd
from IPython.utils import io

nltk.download('punkt')


class DM_predict(object):
    def __init__(self, dm_trained_model, exclude_attrs, one_by_one=False):
        self.dm = dm_trained_model
        self.exclude_attrs = exclude_attrs
        self.one_by_one = one_by_one

    def predict(self, dataset):
        """

        Args:
            dataset: dataset to be predicted wih the same structure of the training dataset

        Returns: list of match scores

        """

        with io.capture_output() as captured:
            self.predictions = []
            file_path = f'/tmp/candidate.csv'
            if self.one_by_one:
                for i in range(dataset.shape[0]):
                    dataset.iloc[[i]].to_csv(file_path, index_label='id')  # /tmp/ path for colab env

                    candidate = dm.data.process_unlabeled(path=file_path, trained_model=self.dm,
                                                          ignore_columns=self.exclude_attrs)
                    self.predictions += [self.dm.run_prediction(candidate)]
                os.remove(file_path)
                res = pd.concat(self.predictions)['match_score'].values
            else:
                dataset.to_csv(file_path, index_label='id')  # /tmp/ path for colab env
                candidate = dm.data.process_unlabeled(path=file_path, trained_model=self.dm,
                                                      ignore_columns=self.exclude_attrs)
                self.predictions = self.dm.run_prediction(candidate)
                os.remove(file_path)
                res = self.predictions['match_score'].values
        return res

