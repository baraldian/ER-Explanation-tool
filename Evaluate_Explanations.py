import re

import numpy as np
import pandas as pd

from LIME_ER_Wrapper import LIME_ER_Wrapper


class Evaluate_explanation(LIME_ER_Wrapper):

    def __init__(self, impacts_df, dataset, predict_proba, exclude_attrs=['id', 'label'], split_expression=' ',
                 select_tokens_func=None, percentage=.25, num_round=10, lprefix='left_', rprefix='right_'):
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.impacts_df = impacts_df
        self.dataset = dataset
        self.cols = [x for x in dataset.columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]

        self.select_tokens_func = self.get_tokens_to_remove if select_tokens_func is None else select_tokens_func
        self.percentage = percentage
        self.num_round = num_round
        self.split_expression = split_expression
        self.splitter = re.compile(split_expression)
        self.predict_proba = predict_proba

    def evaluate_set(self, ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation=None,
                     add_after_perturbation=None, overlap=True):
        impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
        res = []
        if variable_side == 'all':
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.lprefix)]

        for id in ids:
            impact_df = impacts_all[impacts_all.id == id][['word_prefix', 'impact']]
            start_el = self.dataset[self.dataset.id == id]
            res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
                                         add_after_perturbation, overlap)

        if variable_side == 'all':
            impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.rprefix)]
            for id in ids:
                impact_df = impacts_all[impacts_all.id == id][['word_prefix', 'impact']]
                start_el = self.dataset[self.dataset.id == id]
                res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)

        res_df = pd.DataFrame(res)
        res_df['conf'] = conf_name
        res_df['error'] = res_df.expected_delta - res_df.detected_delta
        return res_df

    def evaluate_impacts(self, start_el, impacts_df, variable_side='left', fixed_side='right',
                         add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True):

        impacts_sorted = impacts_df.sort_values('impact', ascending=False)
        self.words_with_prefixes = impacts_sorted['word_prefix'].values
        self.impacts = impacts_sorted['impact'].values

        self.variable_encoded = self.prepare_element(start_el.copy(), variable_side, fixed_side,
                                                     add_before_perturbation, add_after_perturbation, overlap)

        self.start_pred = self.predict_proba(self.restructure_strings([self.variable_encoded]))[0]

        # assert start_pred > .5
        evaluation = {'id': start_el.id.values[0], 'start_pred': self.start_pred}
        # remove tokens --> start from class 1 going to 0

        res_list = []
        combinations_el = {}
        combinations_to_remove = self.select_tokens_func(self.start_pred, self.words_with_prefixes, self.impacts)
        # {'firtsK': [[0], [0, 1, 2], [0, 1, 2, 3, 4]], 'random': [array([ 6, 15, 25, 11, 31, 24, 23,  4]),...]}
        description_to_evaluate = []
        comb_name_sequence = []
        tokens_to_remove_sequence = []
        for comb_name, combinations in combinations_to_remove.items():
            for tokens_to_remove in combinations:
                tmp_encoded = self.variable_encoded
                for token_with_prefix in self.words_with_prefixes[tokens_to_remove]:
                    tmp_encoded = tmp_encoded.replace(str(token_with_prefix), '')
                description_to_evaluate.append(tmp_encoded)
                comb_name_sequence.append(comb_name)
                tokens_to_remove_sequence.append(tokens_to_remove)

        self.perturbed_elements = self.restructure_strings(description_to_evaluate)

        preds = self.predict_proba(self.perturbed_elements)
        for new_pred, tokens_to_remove, comb_name in zip(preds, tokens_to_remove_sequence, comb_name_sequence):
            correct = (new_pred > .5) == (self.start_pred - np.sum(self.impacts[tokens_to_remove]) > .5)
            evaluation.update(comb_name=comb_name, new_pred=new_pred, correct=correct,
                              expected_delta=np.sum(self.impacts[tokens_to_remove]),
                              detected_delta=-(new_pred - self.start_pred),
                              tokens_removed=list(self.words_with_prefixes[tokens_to_remove]))
            res_list.append(evaluation.copy())
        return res_list

    def get_tokens_to_remove(self, start_pred, tokens_sorted, impacts_sorted):
        if len(tokens_sorted) >= 5:
            combination = {'firts1': [[0]], 'first2': [[0, 1]], 'first5': [[0, 1, 2, 3, 4]]}
        else:
            combination = {'firts1': [[0]]}
        i = 1
        tokens_to_remove = [0]
        while len(tokens_to_remove) < len(impacts_sorted) and start_pred - np.sum(
                impacts_sorted[tokens_to_remove]) > 0.5:
            tokens_to_remove.append(i)
            i += 1
        combination['changeClass'] = [tokens_to_remove]
        lent = len(tokens_sorted)
        ntokens = int(lent * self.percentage)
        combination['random'] = [np.random.choice(lent, ntokens) for i in range(self.num_round)]
        return combination

    def generate_evaluation(self, ids, fixed: str, overlap=True):
        evaluation_res = {}
        if fixed == 'right':
            fixed, f = 'right', 'R'
            variable, v = 'left', 'L'
        elif fixed == 'left':
            fixed, f = 'left', 'L'
            variable, v = 'right', 'R'
        else:
            assert False
        ov = '' if overlap == True else 'NOV'

        conf_name = f'{f}_{v}+{f}before{ov}'
        res_df = self.evaluate_set(ids, conf_name, fixed_side=fixed, variable_side=variable,
                                   add_before_perturbation=fixed, overlap=overlap)
        evaluation_res[conf_name] = res_df

        conf_name = f'{f}_{f}+{v}after{ov}'
        res_df = self.evaluate_set(ids, conf_name, fixed_side=fixed, variable_side=fixed,
                                   add_after_perturbation=variable,
                                   overlap=overlap)
        evaluation_res[conf_name] = res_df

        return evaluation_res

    def evaluation_routine(self, ids):

        evaluations_dict = self.generate_evaluation(ids, fixed='right', overlap=True)
        evaluations_dict.update(self.generate_evaluation(ids, fixed='right', overlap=False))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=True))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=False))
        res_df = self.evaluate_set(ids, 'all', variable_side='all')
        evaluations_dict['all'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_R', variable_side='right', fixed_side='left', add_after_perturbation='right')
        evaluations_dict['mojito_copy_R'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_L', variable_side='left', fixed_side='right', add_after_perturbation='left')
        evaluations_dict['mojito_copy_L'] = res_df

        return pd.concat(list(evaluations_dict.values()))
