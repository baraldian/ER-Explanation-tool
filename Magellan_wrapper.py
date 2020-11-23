from Wrapper_functions import map_word_to_attr
import re
import numpy as np
import pandas as pd
import py_entitymatching as em
from IPython.utils import io

class MG_predictor(object):
    def __init__(self, model, feature_table, exclude_attrs):
        self.model = model
        self.exclude_attrs = exclude_attrs
        self.feature_table = feature_table

    def predict(self, dataset):

        with io.capture_output() as captured:
            dataset['id'] = dataset['left_id'] = dataset['right_id'] = np.arange(dataset.shape[0])
            leftDF = dataset[[x for x in dataset.columns if x.startswith('left_')]].copy()
            leftDF.columns = [x[len('left_'):] for x in leftDF.columns]
            rightDF = dataset[[x for x in dataset.columns if x.startswith('right_')]].copy()
            rightDF.columns = [x[len('right_'):] for x in rightDF.columns]

            em.set_key(dataset, 'id')
            em.set_key(leftDF, 'id')
            em.set_key(rightDF, 'id')
            em.set_ltable(dataset, leftDF)
            em.set_rtable(dataset, rightDF)
            em.set_fk_ltable(dataset, 'left_id')
            em.set_fk_rtable(dataset, 'right_id')

            exctracted_features = em.extract_feature_vecs(dataset, feature_table=self.feature_table)
            exctracted_features = em.impute_table(exctracted_features, strategy='mean')
            exclude_tmp = list(set(self.exclude_attrs) - (set(self.exclude_attrs) - set(exctracted_features.columns)))
            self.predictions = self.model.predict(table=exctracted_features, exclude_attrs=exclude_tmp, return_probs=True,
                                                  target_attr='pred', probs_attr='match_score', append=True)
        return self.predictions['match_score'].values


class Magellan_predict_wrapper(object):
    def __init__(self, model, dataset, exclude_attrs, Feature_Table, variable_side='left'):
        self.exclude_attrs = exclude_attrs
        self.model = model
        assert variable_side == 'left' or variable_side == 'right', 'variable_side must be "left" or "right"'
        self.variable_side = variable_side
        self.fixed_side = 'right' if variable_side == 'left' else 'left'
        self.dataset = dataset
        self.variable_cols = [x for x in dataset.columns if
                              x.startswith(variable_side + '_') and x not in exclude_attrs]
        self.fixed_cols = [x.replace(variable_side, self.fixed_side) for x in self.variable_cols]
        self.splitter = re.compile(r'\W+')
        self.Feature_Table = Feature_Table

    def set_fixed_tuple(self, idx):
        self.idx = idx
        self.el = self.dataset.iloc[[idx]]
        self.col_words = {col: self.splitter.split(str(self.el[col].values[0])) for col in self.variable_cols}
        self.fixed_data = self.el[self.fixed_cols]

    def predict_proba(self, variable_data):
        """ Il metodo predict_proba deve prendere in input una lista di
          un elemento contenente le parole da variare per l'analisi del modello
        """
        left_df = map_word_to_attr(variable_data[0], self.col_words, self.variable_cols)
        for single_row in variable_data[1:]:
            tmp_df = map_word_to_attr(single_row, self.col_words, self.variable_cols)
            left_df = pd.concat([left_df, tmp_df])
        left_df = left_df.reset_index()

        right_df = self.fixed_data.loc[[self.idx] * left_df.shape[0]]
        right_df.reset_index(inplace=True, drop=True)
        tmp = pd.concat([left_df, right_df], axis=1)

        tmpJ = self.magellan_feature_extraction(tmp)
        exclude_tmp = list(set(self.exclude_attrs) - (set(self.exclude_attrs) - set(tmpJ.columns)))
        predictions = self.model.predict(table=tmpJ, exclude_attrs=exclude_tmp, return_probs=True, target_attr='pred',
                                         probs_attr='match_score')
        ret = np.ndarray(shape=(predictions[1].shape[0], 2))
        ret[:, 1] = predictions[1]
        ret[:, 0] = 1 - predictions[1]
        return ret

    def magellan_feature_extraction(self, tmp):
        tmp['id'] = tmp['left_id'] = tmp['right_id'] = np.arange(tmp.shape[0])
        leftDF = tmp[[x for x in tmp.columns if x.startswith('left_')]].copy()
        leftDF.columns = [x[len('left_'):] for x in leftDF.columns]
        rightDF = tmp[[x for x in tmp.columns if x.startswith('right_')]].copy()
        rightDF.columns = [x[len('right_'):] for x in rightDF.columns]

        em.set_key(tmp, 'id')
        em.set_key(leftDF, 'id')
        em.set_key(rightDF, 'id')
        em.set_ltable(tmp, leftDF)
        em.set_rtable(tmp, rightDF)
        em.set_fk_ltable(tmp, 'left_id')
        em.set_fk_rtable(tmp, 'right_id')

        tmpH = em.extract_feature_vecs(tmp, feature_table=self.Feature_Table)
        tmpJ = em.impute_table(tmpH, strategy='mean')
        return tmpJ