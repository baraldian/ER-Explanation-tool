import gc

import numpy as np
import py_entitymatching as em
from IPython.utils import io


class MG_predictor(object):
    def __init__(self, model, feature_table, exclude_attrs, lprefix='left_', rprefix='right_'):
        self.model = model
        self.exclude_attrs = exclude_attrs
        self.feature_table = feature_table
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.columns = list(feature_table.left_attribute.unique()) + ['id']
        self.lcolumns = [lprefix + col for col in self.columns]
        self.rcolumns = [rprefix + col for col in self.columns]
        

    def predict(self, dataset, impute_value=0):
        dataset = dataset.copy()
        with io.capture_output() as captured:
            dataset['id'] = dataset['left_id'] = dataset['right_id'] = np.arange(dataset.shape[0])
            leftDF = dataset[self.lcolumns].copy()
            leftDF.columns = self.columns
            rightDF = dataset[self.rcolumns].copy()
            rightDF.columns = self.columns

            em.set_key(dataset, 'id')
            em.set_key(leftDF, 'id')
            em.set_key(rightDF, 'id')
            em.set_ltable(dataset, leftDF)
            em.set_rtable(dataset, rightDF)
            em.set_fk_ltable(dataset, 'left_id')
            em.set_fk_rtable(dataset, 'right_id')

            self.exctracted_features = em.extract_feature_vecs(dataset, feature_table=self.feature_table)
            self.exctracted_features = self.exctracted_features.fillna(impute_value)
            exclude_tmp = list(set(self.exclude_attrs) - (set(self.exclude_attrs) - set(self.exctracted_features.columns)))
            self.predictions = self.model.predict(table=self.exctracted_features, exclude_attrs=exclude_tmp, return_probs=True,
                                                  target_attr='pred', probs_attr='match_score', append=True)
        del dataset
        gc.collect()
        return self.predictions['match_score'].values
