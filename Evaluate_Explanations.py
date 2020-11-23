import numpy as np
import pandas as pd

from LIME_ER_Wrapper import Mapper


class Evaluate_explanation():

    def __init__(self, columns, exclude_attrs=['id', 'label'], predict_proba,
                         split_expression=' ', select_tokens_func=None, percentage=.25, num_round=10):


        self.cols = [x for x in columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]

        self.select_tokens_func = self.get_tokens_to_remove if select_tokens_func is None else select_tokens_func
        self.percentage = percentage
        self.num_round = num_round
        self.split_expression = split_expression
        self.predict_proba = predict_proba

    def evaluate_impacts(self, start_el, impacts_df, variable_side='left', fixed_side='right', add_before_perturbation=None,
                         add_after_perturbation=None ):


        self.start_pred = self.predict_proba(start_el)[0]
        impacts_sorted = impacts_df.sort_values('impact', ascending=False)
        self.tokens = impacts_sorted['word_prefix'].values
        self.impacts = impacts_sorted['impact'].values
        self.fixed_el = start_el[self.fixed_cols].reset_index(drop=True) if self.fixed_cols != [] else None
        self.mapper = Mapper(self.variable_cols, split_expression=self.split_expression)

        self.variable_encoded = self.mapper.encode_attr(start_el)

        # assert start_pred > .5
        evaluation = {'id': start_el.id.values[0], 'start_pred': self.start_pred}
        # remove tokens --> start from class 1 going to 0

        res_list = []
        combinations_el = {}
        combinations_to_remove = self.select_tokens_func(self.start_pred, self.tokens, self.impacts)
        # {'firtsK': [[0], [0, 1, 2], [0, 1, 2, 3, 4]], 'random': [array([ 6, 15, 25, 11, 31, 24, 23,  4]),...]}
        description_to_evaluate = []
        comb_name_sequence = []
        tokens_to_remove_sequence =[]
        for comb_name, combinations in combinations_to_remove.items():
            for tokens_to_remove in combinations:
                tmp_encoded = self.variable_encoded
                for token_with_prefix in self.tokens[tokens_to_remove]:
                    tmp_encoded = tmp_encoded.replace(str(token_with_prefix), '')
                description_to_evaluate.append(tmp_encoded)
                comb_name_sequence.append(comb_name)
                tokens_to_remove_sequence.append(tokens_to_remove)

        perturbed_elements = self.restructure_strings(description_to_evaluate)
        preds = self.predict_proba(perturbed_elements)
        for new_pred, tokens_to_remove, comb_name in zip(preds, tokens_to_remove_sequence, comb_name_sequence):
            correct =  (new_pred > .5) == (self.start_pred - np.sum(self.impacts[tokens_to_remove]) > .5)
            evaluation.update(comb_name=comb_name, new_pred=new_pred, correct=correct,
                              expected_delta=np.sum(self.impacts[tokens_to_remove]),
                              detected_delta=-(new_pred - self.start_pred),
                              tokens_removed=[list(self.tokens[tokens_to_remove])])
            res_list.append(evaluation.copy())
        return res_list

    def get_tokens_to_remove(self, start_pred, tokens_sorted, impacts_sorted):
        combination = {'firtsK': [[0], [0, 1, 2], [0, 1, 2, 3, 4]]}
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

    def restructure_strings(self, perturbed_strings):
        df_list = []
        for single_row in perturbed_strings:
            df_list.append(self.mapper.map_word_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.fixed_el is not None:
            fixed_df = pd.concat([self.fixed_el] * variable_df.shape[0])
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)
