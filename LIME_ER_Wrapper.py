import re

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer


class Mapper(object):
    def __init__(self, columns, split_expression):
        self.columns = columns
        self.attr_map = {chr(ord('A') + colidx): col for colidx, col in enumerate(self.columns)}
        self.arange = np.arange(100)
        self.split_expression = split_expression

    def map_word_to_attr(self, text_to_restructure):
        res = re.findall(r'(?P<attr>[A-Z]{1})(?P<pos>[0-9]{2})_(?P<word>[^' + self.split_expression + ']+)',
                         text_to_restructure)
        structured_row = {col: '' for col in self.columns}
        for col_code, pos, word in res:
            structured_row[self.attr_map[col_code]] += word + ' '
        for col in self.columns: # Remove last space
            structured_row[col] = structured_row[col][:-1]
        return structured_row

    def encode_attr(self, el):
        return ' '.join(
            [chr(ord('A') + colpos) + "{:02d}_".format(wordpos) + word for colpos, col in enumerate(self.columns) for
             wordpos, word in enumerate(re.split(self.split_expression, str(el[col].values[0])))])


class LIME_ER_Wrapper(object):

    def __init__(self, predict_method, dataset, exclude_attrs=['id', 'label'], split_expression=r'\W+',
                 lprefix='ltable_', rprefix='rtable_', **argv, ):
        self.splitter = re.compile(split_expression)
        self.split_expression = split_expression
        self.explainer = LimeTextExplainer(class_names=['NO match', 'MATCH'], split_expression=split_expression, **argv)
        self.model_predict = predict_method
        self.dataset = dataset
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.exclude_attrs = exclude_attrs

        self.cols = [x for x in dataset.columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]
        self.explanations = {}

    def explain(self, elements, kind='all', **argv, ):
        feasible_values = ['all', 'left', 'right']
        assert kind in feasible_values, f"kind must have a value between {feasible_values}"

        left_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='left', fixed_side='right', **argv)
            impacts['side'] = 'left'
            left_impacts.append(impacts)

        right_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='right', fixed_side='left', **argv)
            impacts['side'] = 'right'
            right_impacts.append(impacts)

        total_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='all', fixed_side=None, **argv)
            impacts['side'] = 'all'
            total_impacts.append(impacts)
        self.impacts = pd.concat(left_impacts + right_impacts + total_impacts)
        return self.impacts

    def explain_instance(self, el, variable_side='left', fixed_side='right', add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True, **argv):
        # assert variable_side in ['left', 'right', 'all'], f'variable_side must be "left" or "right", found: {variable_side}'
        self.add_after_perturbation = add_after_perturbation
        self.overlap = overlap
        variable_el = el.copy()

        if variable_side in ['left', 'right']:
            variable_cols = self.left_cols if variable_side == 'left' else self.right_cols
            if add_before_perturbation is not None or add_after_perturbation is not None:
                self.compute_tokens(el)
                if add_before_perturbation is not None:
                    self.add_tokens(variable_el, variable_cols, add_before_perturbation, overlap)
            variable_data = Mapper(variable_cols, self.split_expression).encode_attr(variable_el)

            assert fixed_side in ['left', 'right']
            if fixed_side == 'left':
                fixed_cols, not_fixed_cols = self.left_cols, self.right_cols
            else:
                fixed_cols, not_fixed_cols = self.right_cols, self.left_cols
            mapper_fixed = Mapper(fixed_cols, self.split_expression)
            self.fixed_data = pd.DataFrame([mapper_fixed.map_word_to_attr(mapper_fixed.encode_attr(
                el[fixed_cols]))])  # encode and decode data of fixed source to ensure the same format
            self.mapper_variable = Mapper(not_fixed_cols, self.split_expression)

        elif variable_side == 'all':
            variable_cols = self.cols
            self.mapper_variable = Mapper(variable_cols, self.split_expression)
            self.fixed_data = None
            fixed_side = 'all'
            variable_data =self.mapper_variable.encode_attr(variable_el)
        else:
            assert False, f'Not a feasible configuration. variable_side: {variable_side} not allowed.'


        words = self.splitter.split(variable_data)
        explanation = self.explainer.explain_instance(variable_data, self.predict_proba, num_features=len(words),
                                                      **argv)
        self.variable_data = variable_data # to test the addition before perturbation

        id = el.id.values[0]  # Assume index is the id column
        self.explanations[f'{fixed_side}{id}'] = explanation
        return self.explanation_to_df(explanation, words, self.mapper_variable.attr_map, id)

    def explanation_to_df(self, explanation, words, attribute_map, id):
        impacts_list = []
        dict_impact = {'id': id}
        for wordpos, impact in explanation.as_map()[1]:
            word = words[wordpos]
            dict_impact.update(column=attribute_map[word[0]], position=int(word[1:3]), word=word[4:], word_prefix=word,
                               impact=impact)
            impacts_list.append(dict_impact.copy())

        return pd.DataFrame(impacts_list)

    def compute_tokens(self, el):
        tokens = {col: np.array(self.splitter.split(str(el[col].values[0]))) for col in self.cols}
        tokens_intersection = {}
        tokens_not_overlapped = {}
        for col in [col.replace('left_', '') for col in self.left_cols]:
            lcol, rcol = self.lprefix + col, self.rprefix + col

            tokens_intersection[col] = np.intersect1d(tokens[lcol], tokens[rcol])
            tokens_not_overlapped[lcol] = tokens[lcol][~ np.in1d(tokens[lcol], tokens_intersection[col])]
            tokens_not_overlapped[rcol] = tokens[rcol][~ np.in1d(tokens[rcol], tokens_intersection[col])]
        self.tokens_not_overlapped = tokens_not_overlapped
        self.tokens_intersection = tokens_intersection
        self.tokens = tokens

    def add_tokens(self, el, dst_columns, src_side, overlap=True):
        if overlap == False:
            add_tokens = self.tokens_not_overlapped
        else:
            add_tokens = self.tokens

        if src_side == 'left':
            src_columns = self.left_cols
        elif src_side == 'right':
            src_columns = self.right_cols
        else:
            assert False, f'src_side must "left" or "right". Got {src_side}'

        for col_dst, col_src in zip(dst_columns, src_columns):
            el[col_dst] = el[col_dst] + ' ' + ' '.join(add_tokens[col_src])

    def predict_proba(self, perturbed_strings):
        """ Il metodo predict_proba deve prendere in input una lista di
        un elemento contenente le parole da variare per l'analisi del modello
        """

        df_list = [self.mapper_variable.map_word_to_attr(perturbed_strings[0])]
        for single_row in perturbed_strings[1:]:
            df_list.append(self.mapper_variable.map_word_to_attr(single_row))
        left_df = pd.DataFrame.from_dict(df_list)

        if self.add_after_perturbation is not None:
            self.add_tokens(left_df, left_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            right_df = pd.concat([self.fixed_data] * left_df.shape[0])
            right_df.reset_index(inplace=True, drop=True)
        else:
            right_df = None

        self.tmp_dataset = pd.concat([left_df, right_df], axis=1)
        self.tmp_dataset.reset_index(inplace=True, drop=True)
        predictions = self.model_predict(self.tmp_dataset)

        ret = np.ndarray(shape=(len(predictions), 2))
        ret[:, 1] = np.array(predictions)
        ret[:, 0] = 1 - ret[:, 1]
        return ret



