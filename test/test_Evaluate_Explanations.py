from unittest import TestCase

import numpy as np
import pandas as pd

from Evaluate_Explanations import Evaluate_explanation
from LIME_ER_Wrapper import LIME_ER_Wrapper


class Test(TestCase):
    def setUp(self) -> None:
        self.fake_pred = lambda x: np.ones((x.shape[0],)) * 0.5

        def randomf(x):
            return np.random.rand(x.shape[0])

        self.random_pred = randomf

    def test_evaluate_set(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2', 'm1 m2', 'r1', 's1 s2 s3'
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.random_pred, el, exclude_attrs=[], lprefix='left_',
                                    rprefix='right_', split_expression=r' ')
        num_sample = 5
        impacts_match = explainer.explain(el, num_samples=num_sample)

        ev = Evaluate_explanation(impacts_match, el, self.random_pred, el.columns,
                                  percentage=.25, num_round=5)
        results = ev.evaluate_set([1], 'all', variable_side='all')

        encoded = 'A00_l1 A01_l2 B00_m1 B01_m2 C00_r1 D00_s1 D01_s2 D02_s3'
        self.assertEqual(ev.variable_encoded, encoded)
        self.assertEqual(ev.fixed_data, None)
        self.assertEqual(results.id.unique(), [1])

    def test_Evaluate_Right_Right_addAFTERLeft(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2'
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        impacts_match = explainer.explain_instance(el, variable_side='right', fixed_side='right',
                                                   add_after_perturbation='left', num_samples=5)
        conf_name = 'R_R+Lafter'
        impacts_match['conf'] = conf_name

        ev = Evaluate_explanation(impacts_match, el, self.random_pred, el.columns,
                                  percentage=.25, num_round=5)
        results = ev.evaluate_set([1], conf_name, variable_side='right', fixed_side='right',
                                  add_after_perturbation='left')
        self.assertTrue(ev.perturbed_elements.left_A.str.endswith(lstring1).all())
        self.assertTrue(ev.perturbed_elements.left_B.str.endswith(lstring2).all())
        encoded = 'A00_r1 A01_r2 A02_r3 B00_s1 B01_s2'
        self.assertEqual(ev.variable_encoded, encoded)
        self.assertTrue(ev.fixed_data.equals(el[[x for x in el.columns if x.startswith('right_')]]))

    def test_Evaluate_Right_Right_addBEFORELeft(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2', 'r1 r2 r3', 's1 s2'
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        impacts_match = explainer.explain_instance(el, variable_side='right', fixed_side='right',
                                                   add_before_perturbation='left', num_samples=5)
        conf_name = 'R_R+Lafter'
        impacts_match['conf'] = conf_name

        ev = Evaluate_explanation(impacts_match, el, self.random_pred, el.columns,
                                  percentage=.25, num_round=5)
        results = ev.evaluate_set([1], conf_name, variable_side='right', fixed_side='right',
                                  add_before_perturbation='left')
        encoded = 'A00_r1 A01_r2 A02_r3 A03_l1 A04_l2 A05_l3 A06_l1 A07_l5 B00_s1 B01_s2 B02_m1 B03_m2'
        self.assertEqual(ev.variable_encoded, encoded)
        self.assertTrue(ev.fixed_data.equals(el[[x for x in el.columns if x.startswith('right_')]]))

    def test_Evaluate_Right_Left_addBEFORELeft(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2', 'r1 r2 r3', 's1 s2'
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        impacts_match = explainer.explain_instance(el, variable_side='right', fixed_side='left',
                                                   add_before_perturbation='left', num_samples=5)
        conf_name = 'R_R+Lafter'
        impacts_match['conf'] = conf_name

        ev = Evaluate_explanation(impacts_match, el, self.random_pred, el.columns,
                                  percentage=.25, num_round=5)
        results = ev.evaluate_set([1], conf_name, variable_side='right', fixed_side='left',
                                  add_before_perturbation='left')
        encoded = 'A00_r1 A01_r2 A02_r3 A03_l1 A04_l2 A05_l3 A06_l1 A07_l5 B00_s1 B01_s2 B02_m1 B03_m2'
        self.assertEqual(ev.variable_encoded, encoded)
        self.assertTrue(ev.fixed_data.equals(el[[x for x in el.columns if x.startswith('left_')]]))
