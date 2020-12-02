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

    def map_word_to_attr_dict(self, text_to_restructure):
        res = re.findall(r'(?P<attr>[A-Z]{1})(?P<pos>[0-9]{2})_(?P<word>[^' + self.split_expression + ']+)',
                         text_to_restructure)
        structured_row = {col: '' for col in self.columns}
        for col_code, pos, word in res:
            structured_row[self.attr_map[col_code]] += word + ' '
        for col in self.columns:  # Remove last space
            structured_row[col] = structured_row[col][:-1]
        return structured_row

    def map_word_to_attr(self, text_to_restructure):
        return pd.DataFrame([self.map_word_to_attr_dict(text_to_restructure)])

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
        self.cols = self.left_cols + self.right_cols
        self.explanations = {}

    def explain(self, elements, kind='all', **argv, ):
        feasible_values = ['all', 'left', 'right']
        assert kind in feasible_values, f"kind must have a value between {feasible_values}"

        left_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='left', fixed_side='right', **argv)
            impacts['conf'] = 'left'
            left_impacts.append(impacts)

        right_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='right', fixed_side='left', **argv)
            impacts['conf'] = 'right'
            right_impacts.append(impacts)

        total_impacts = []
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side='all', fixed_side=None, **argv)
            impacts['conf'] = 'all'
            total_impacts.append(impacts)
        self.impacts = pd.concat(left_impacts + right_impacts + total_impacts)
        return self.impacts

    def explain_instance(self, el, variable_side='left', fixed_side='right', add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True, **argv):
        variable_el = el.copy()
        for col in self.cols:
            variable_el[col] = ' '.join(re.split(r' +', str(variable_el[col].values[0]).strip()))

        variable_data = self.prepare_element(variable_el, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)

        words = self.splitter.split(variable_data)
        explanation = self.explainer.explain_instance(variable_data, self.predict_proba, num_features=len(words),
                                                      **argv)
        self.variable_data = variable_data  # to test the addition before perturbation

        id = el.id.values[0]  # Assume index is the id column
        self.explanations[f'{self.fixed_side}{id}'] = explanation
        return self.explanation_to_df(explanation, words, self.mapper_variable.attr_map, id)

    def prepare_element(self, variable_el, variable_side, fixed_side, add_before_perturbation, add_after_perturbation,
                        overlap):
        """
        Set fixed_side, fixed_data, mapper_variable.
        Call compute_tokens if needed

        """
        self.add_after_perturbation = add_after_perturbation
        self.overlap = overlap
        self.fixed_side = fixed_side
        if variable_side in ['left', 'right']:
            variable_cols = self.left_cols if variable_side == 'left' else self.right_cols

            assert fixed_side in ['left', 'right']
            if fixed_side == 'left':
                fixed_cols, not_fixed_cols = self.left_cols, self.right_cols
            else:
                fixed_cols, not_fixed_cols = self.right_cols, self.left_cols
            mapper_fixed = Mapper(fixed_cols, self.split_expression)
            self.fixed_data = mapper_fixed.map_word_to_attr(mapper_fixed.encode_attr(
                variable_el[fixed_cols]))  # encode and decode data of fixed source to ensure the same format
            self.mapper_variable = Mapper(not_fixed_cols, self.split_expression)

            if add_before_perturbation is not None or add_after_perturbation is not None:
                self.compute_tokens(variable_el)
                if add_before_perturbation is not None:
                    self.add_tokens(variable_el, variable_cols, add_before_perturbation, overlap)
            variable_data = Mapper(variable_cols, self.split_expression).encode_attr(variable_el)

        elif variable_side == 'all':
            variable_cols = self.left_cols + self.right_cols
            self.mapper_variable = Mapper(variable_cols, self.split_expression)
            self.fixed_data = None
            self.fixed_side = 'all'
            variable_data = self.mapper_variable.encode_attr(variable_el)
        else:
            assert False, f'Not a feasible configuration. variable_side: {variable_side} not allowed.'
        return variable_data

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
            tokens_to_add = self.tokens_not_overlapped
        else:
            tokens_to_add = self.tokens

        if src_side == 'left':
            src_columns = self.left_cols
        elif src_side == 'right':
            src_columns = self.right_cols
        else:
            assert False, f'src_side must "left" or "right". Got {src_side}'

        for col_dst, col_src in zip(dst_columns, src_columns):
            el[col_dst] = el[col_dst].astype(str) + ' ' + ' '.join(tokens_to_add[col_src])

    def predict_proba(self, perturbed_strings):
        """ Il metodo predict_proba deve prendere in input una lista di
        un elemento contenente le parole da variare per l'analisi del modello
        """
        self.tmp_dataset = self.restructure_strings(perturbed_strings)
        self.tmp_dataset.reset_index(inplace=True, drop=True)
        predictions = self.model_predict(self.tmp_dataset)

        ret = np.ndarray(shape=(len(predictions), 2))
        ret[:, 1] = np.array(predictions)
        ret[:, 0] = 1 - ret[:, 1]
        return ret

    def restructure_strings(self, perturbed_strings):
        df_list = []
        for single_row in perturbed_strings:
            df_list.append(self.mapper_variable.map_word_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.add_after_perturbation is not None:
            self.add_tokens(variable_df, variable_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            fixed_df = pd.concat([self.fixed_data] * variable_df.shape[0])
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)

    def generate_explanation(self, el, explainer, fixed: str, num_sample=1000, overlap=True):
        explanations_df = []
        if fixed == 'right':
            fixed, f = 'right', 'R'
            variable, v = 'left', 'L'
        elif fixed == 'left':
            fixed, f = 'left', 'L'
            variable, v = 'right', 'R'
        else:
            assert False
        ov = '' if overlap == True else 'NOV'

        tmp = self.explain_instance(el, fixed_side=fixed, variable_side=variable, add_after_perturbation=fixed,
                                         num_samples=num_sample, overlap=overlap)
        tmp['conf'] = f'{f}_{v}+{f}after{ov}'
        explanations_df.append(tmp)

        tmp = self.explain_instance(el, fixed_side=fixed, variable_side=variable, add_before_perturbation=fixed,
                                         num_samples=num_sample, overlap=overlap)
        tmp['conf'] = f'{f}_{v}+{f}before{ov}'
        explanations_df.append(tmp)

        tmp = self.explain_instance(el, fixed_side=fixed, variable_side=fixed, add_after_perturbation=variable,
                                         num_samples=num_sample, overlap=overlap)
        tmp['conf'] = f'{f}_{f}+{v}after{ov}'
        explanations_df.append(tmp)
        return explanations_df

    def explanation_routine(self, el, explainer, num_sample=1000):
        explanations_df = []
        tmp = self.explain_instance(el, variable_side='all', fixed_side=None, num_samples=num_sample)
        tmp['conf'] = 'all'

        explanations_df.append(tmp)
        explanations_df += self.generate_explanation(el, explainer, fixed='right', num_sample=num_sample, overlap=True)
        explanations_df += self.generate_explanation(el, explainer, fixed='right', num_sample=num_sample, overlap=False)
        explanations_df += self.generate_explanation(el, explainer, fixed='left', num_sample=num_sample, overlap=True)
        explanations_df += self.generate_explanation(el, explainer, fixed='left', num_sample=num_sample, overlap=False)
        return pd.concat(explanations_df)
