import numpy as np
import pandas as pd

from LIME_ER_Wrapper import LIME_ER_Wrapper


class Evaluate_explanation(LIME_ER_Wrapper):

    def __init__(self, impacts_df, dataset, select_tokens_func=None, percentage=.25, num_round=10, **argv):
        self.impacts_df = impacts_df

        self.select_tokens_func = self.get_tokens_to_remove if select_tokens_func is None else select_tokens_func
        self.percentage = percentage
        self.num_round = num_round
        super().__init__(dataset=dataset, **argv)

    def prepare_impacts(self, impacts_df, start_el, variable_side, fixed_side,
                        add_before_perturbation, add_after_perturbation, overlap):
        impacts_sorted = impacts_df.sort_values('impact', ascending=False)
        self.words_with_prefixes = impacts_sorted['word_prefix'].values
        self.impacts = impacts_sorted['impact'].values

        self.variable_encoded = self.prepare_element(start_el.copy(), variable_side, fixed_side,
                                                     add_before_perturbation, add_after_perturbation, overlap)

        self.start_pred = self.predict_proba([self.variable_encoded])[:, 1][0]  # match_score

    def generate_descriptions(self, combinations_to_remove):
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
        return description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence

    def evaluate_impacts(self, start_el, impacts_df, variable_side='left', fixed_side='right',
                         add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True):

        self.prepare_impacts(impacts_df, start_el, variable_side, fixed_side, add_before_perturbation,
                             add_after_perturbation, overlap)

        evaluation = {'id': start_el.id.values[0], 'start_pred': self.start_pred}

        res_list = []

        combinations_to_remove = self.select_tokens_func(self.start_pred, self.words_with_prefixes, self.impacts)
        # {'firtsK': [[0], [0, 1, 2], [0, 1, 2, 3, 4]], 'random': [array([ 6, 15, 25, 11, 31, 24, 23,  4]),...]}

        description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence = self.generate_descriptions(
            combinations_to_remove)

        preds = self.predict_proba(description_to_evaluate)[:, 1]
        for new_pred, tokens_to_remove, comb_name in zip(preds, tokens_to_remove_sequence, comb_name_sequence):
            correct = (new_pred > .5) == ((self.start_pred - np.sum(self.impacts[tokens_to_remove])) > .5)
            evaluation.update(comb_name=comb_name, new_pred=new_pred, correct=correct,
                              expected_delta=np.sum(self.impacts[tokens_to_remove]),
                              detected_delta=-(new_pred - self.start_pred),
                              tokens_removed=list(self.words_with_prefixes[tokens_to_remove]))
            res_list.append(evaluation.copy())
        return res_list

    def evaluate_utility(self, start_el, impacts_df, variable_side='left', fixed_side='right',
                         add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True):

        self.prepare_impacts(impacts_df, start_el, variable_side, fixed_side, add_before_perturbation,
                             add_after_perturbation, overlap)

        evaluation = {'id': start_el.id.values[0], 'start_pred': self.start_pred}

        res_list = []

        change_class_tokens = self.get_tokens_to_change_class(self.start_pred, self.impacts)
        combinations_to_remove = {'change_class': [change_class_tokens],
                                  'single_word': [[x] for x in np.arange(self.impacts.shape[0])],
                                  'all_opposite': [[pos for pos, impact in enumerate(self.impacts) if
                                                    (impact > 0) == (self.start_pred > .5)]]}

        combinations_to_remove['change_class_D.10'] = [self.get_tokens_to_change_class(self.start_pred, self.impacts, delta=.1)]
        combinations_to_remove['change_class_D.15'] = [self.get_tokens_to_change_class(self.start_pred, self.impacts, delta=.15)]

        description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence = self.generate_descriptions(
            combinations_to_remove)

        preds = self.predict_proba(description_to_evaluate)[:, 1]
        for new_pred, tokens_to_remove, comb_name in zip(preds, tokens_to_remove_sequence, comb_name_sequence):
            correct = (new_pred > .5) == ((self.start_pred - np.sum(self.impacts[tokens_to_remove])) > .5)
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

        tokens_to_remove = self.get_tokens_to_change_class(start_pred, impacts_sorted)
        combination['change_class'] = [tokens_to_remove]
        lent = len(tokens_sorted)
        ntokens = int(lent * self.percentage)
        np.random.seed(0)
        combination['random'] = [np.random.choice(lent, ntokens, ) for i in range(self.num_round)]
        return combination

    def get_tokens_to_change_class(self, start_pred, impacts_sorted, delta=0):
        i = 0
        tokens_to_remove = []
        positive = start_pred > .5
        delta = -delta if not positive else delta
        index = np.arange(0, len(impacts_sorted))
        if not positive:
            index = index[::-1]  # start removing negative impacts to push the score towards match if not positive
        while len(tokens_to_remove) < len(impacts_sorted) and ((start_pred - np.sum(
                impacts_sorted[tokens_to_remove])) > 0.5 + delta) == positive:
            if (impacts_sorted[
                    index[i]] > 0) == positive:  # remove positive impact if element is match, neg impacts if no match
                tokens_to_remove.append(index[i])
                i += 1
            else:
                break
        return tokens_to_remove

    def evaluate_set(self, ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation=None,
                     add_after_perturbation=None, overlap=True, utility=False):
        impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
        res = []
        if variable_side == 'all':
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.lprefix)]

        for id in ids:
            impact_df = impacts_all[impacts_all.id == id][['word_prefix', 'impact']]
            start_el = self.dataset[self.dataset.id == id]
            if utility == True:
                res += self.evaluate_utility(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)
            else:
                res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)

        if variable_side == 'all':
            impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.rprefix)]
            for id in ids:
                impact_df = impacts_all[impacts_all.id == id][['word_prefix', 'impact']]
                start_el = self.dataset[self.dataset.id == id]
                if utility == True:
                    res += self.evaluate_utility(start_el, impact_df, variable_side, fixed_side,
                                                 add_before_perturbation,
                                                 add_after_perturbation, overlap)
                else:
                    res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side,
                                                 add_before_perturbation,
                                                 add_after_perturbation, overlap)

        res_df = pd.DataFrame(res)
        res_df['conf'] = conf_name
        res_df['error'] = res_df.expected_delta - res_df.detected_delta
        return res_df

    def generate_evaluation(self, ids, fixed: str, overlap=True, **argv):
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
                                   add_before_perturbation=fixed, overlap=overlap, **argv)
        evaluation_res[conf_name] = res_df

        """
        conf_name = f'{f}_{f}+{v}after{ov}'
        res_df = self.evaluate_set(ids, conf_name, fixed_side=fixed, variable_side=fixed,
                                   add_after_perturbation=variable,
                                   overlap=overlap, **argv)
        evaluation_res[conf_name] = res_df
        """

        return evaluation_res

    def evaluation_routine(self, ids, **argv):
        assert np.all([x in self.impacts_df.id.unique() and x in self.dataset.id.unique() for x in ids]), \
            f'Missing some explanations {[x for x in ids if x in self.impacts_df.id.unique() or x in self.dataset.id.unique()]}'
        evaluations_dict = self.generate_evaluation(ids, fixed='right', overlap=True, **argv)
        evaluations_dict.update(self.generate_evaluation(ids, fixed='right', overlap=False, **argv))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=True, **argv))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=False, **argv))
        res_df = self.evaluate_set(ids, 'all', variable_side='all', fixed_side=None, **argv)
        evaluations_dict['all'] = res_df
        res_df = self.evaluate_set(ids, 'left', variable_side='left', fixed_side='right', **argv)
        evaluations_dict['left'] = res_df
        res_df = self.evaluate_set(ids, 'right', variable_side='right', fixed_side='left', **argv)
        evaluations_dict['right'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_R', variable_side='right', fixed_side='left', **argv)
        evaluations_dict['mojito_copy_R'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_L', variable_side='left', fixed_side='right', **argv)
        evaluations_dict['mojito_copy_L'] = res_df

        return pd.concat(list(evaluations_dict.values()))
