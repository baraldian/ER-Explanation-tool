import os
import re
from unittest import TestCase

import numpy as np
import pandas as pd

from LIME_ER_Wrapper import LIME_ER_Wrapper


class TestLIME_ER_Wrapper(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_path = 'C:\\Users\\Barald\\UNIdrive\\TESI Baraldi Interpretable ML ER\\datasets'
        dataset_path = os.path.join(dataset_path, 'Abt-Buy')
        cls.data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

        fake_pred = lambda x: np.ones((x.shape[0],)) * 0.5
        proba = fake_pred(cls.data)
        cls.tp_group = cls.data[(1 - proba >= 0.5) & (cls.data['label'] == '1')].head(2)
        cls.tn_group = cls.data[(proba >= 0.5) & (cls.data['label'] == '0')].head(2)
        cls.exclude_attrs = ['left_id', 'right_id', 'label', 'id']

        cls.explainer = LIME_ER_Wrapper(fake_pred, cls.data, exclude_attrs=cls.exclude_attrs, lprefix='left_',
                                        rprefix='right_', split_expression=r' ')
        cls.el = cls.data.iloc[[126]].copy()

    def setUp(self) -> None:
        self.fake_pred = lambda x: np.ones((x.shape[0],)) * 0.5

    def tearDown(self):
        pass

    def test_explain(self):
        # self.fail()
        pass

    def test_Mapper_encode_attr(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l4', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2 s3 s4 s5'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_after_perturbation='right', num_samples=500)
        assert explainer.fixed_data.equals(pd.DataFrame({'right_A': [rstring1], 'right_B': [rstring2]}))
        assert explainer.variable_data == 'A00_l1 A01_l2 A02_l3 A03_l4 B00_m1 B01_m2 B02_m3 B03_m4'

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_',
                                    split_expression=r'\W+')  # Change split expression
        """ The \W metacharacter is used to find a non-word character.
            A word character is a character from a-z, A-Z, 0-9, including the _ (underscore) character."""
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2_.l3 l1^.,l5', 'm1 m2 m3 m4', 'r1 r2-r3', 's1 s2+\'s3 s4 s\|_.,#ù[{5'
        left_string = lstring1 + ' ' + lstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_after_perturbation='right', num_samples=500)
        assert explainer.fixed_data.equals(pd.DataFrame({'right_A': ['r1 r2 r3'], 'right_B': ['s1 s2 s3 s4 s _ ù 5']}))
        assert explainer.variable_data == 'A00_l1 A01_l2_ A02_l3 A03_l1 A04_l5 B00_m1 B01_m2 B02_m3 B03_m4'
        assert len(explainer.explanations['right1'].as_list()) == len(re.split(r'\W+', left_string))

    def test_explain_instance_leftRight_addAFTERRight(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l4', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2 s3 s4 s5'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_after_perturbation='right', num_samples=500)
        self.assertTrue(explainer.tmp_dataset.left_A.str.endswith(rstring1).all())
        self.assertTrue(explainer.tmp_dataset.left_B.str.endswith(rstring2).all())
        assert len(explainer.explanations['right1'].as_list()) == len(left_string.split(' '))

        rstring1, rstring2 = 'r1 r2-r3', 's1 s2+\'s3 s4 s\|_.,#ù[{5'
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_after_perturbation='right', num_samples=500)
        self.assertTrue(explainer.tmp_dataset.left_A.str.endswith(rstring1).all())
        self.assertTrue(explainer.tmp_dataset.left_B.str.endswith(rstring2).all())
        #self.assertEqual(explainer.tmp_dataset.columns, el.columns)
        assert explainer.fixed_data.equals(pd.DataFrame({'right_A': [rstring1], 'right_B': [rstring2]}))
        encoded = 'A00_l1 A01_l2 A02_l3 A03_l4 B00_m1 B01_m2 B02_m3 B03_m4'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['right1'].as_list()], re.split(' ', encoded))

    def test_explain_instance_leftRight_addBEFORERight(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_before_perturbation='right', num_samples=500)
        encoded = 'A00_l1 A01_l2 A02_l3 A03_l1 A04_l5 A05_r1 A06_r2 A07_r3 B00_m1 B01_m2 B02_m3 B03_m4 B04_s1 B05_s2'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['right1'].as_list()], re.split(' ', encoded))

        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l4', 'm1 m2 m3 m4', 'r1 r2-r3 l4', 's1 s2+\'s3 m2'
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_before_perturbation='right', num_samples=500)

        encoded = 'A00_l1 A01_l2 A02_l3 A03_l4 A04_r1 A05_r2-r3 A06_l4 B00_m1 B01_m2 B02_m3 B03_m4 B04_s1 B05_s2+\'s3 B06_m2'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['right1'].as_list()], re.split(' ', encoded))

    def test_explain_instance_leftRight_addBEFORERight_NOoverlap(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2 m3 m4', 'l1 r2 l3 r1', 's1 s2 m2'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='left', fixed_side='right',
                                          add_before_perturbation='right', overlap=False, num_samples=500)
        encoded = 'A00_l1 A01_l2 A02_l3 A03_l1 A04_l5 A05_r2 A06_r1 B00_m1 B01_m2 B02_m3 B03_m4 B04_s1 B05_s2'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['right1'].as_list()], re.split(' ', encoded))


    def test_explain_instance_Right_addLeftAFTER(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l1 l5', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='right', fixed_side='right',
                                          add_after_perturbation='left', num_samples=500)
        self.assertTrue(explainer.tmp_dataset.left_A.str.endswith(lstring1).all())
        self.assertTrue(explainer.tmp_dataset.left_B.str.endswith(lstring2).all())
        encoded = 'A00_r1 A01_r2 A02_r3 B00_s1 B01_s2'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['right1'].as_list()], re.split(' ', encoded))


    def test_explain_instance_ALL(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l4', 'm1 m2 m3 m4', 'r1 r2   r3', 's1 s2 '
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, 0.9, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'match_score', 'left_A', 'left_B', 'right_A', 'right_B'])

        explainer = LIME_ER_Wrapper(self.fake_pred, el, lprefix='left_', rprefix='right_', split_expression=r' ')
        expl = explainer.explain_instance(el, variable_side='all', num_samples=500)
        encoded = 'A00_l1 A01_l2 A02_l3 A03_l4 B00_m1 B01_m2 B02_m3 B03_m4 ' \
                  'C00_r1 C01_r2 C02_r3 D00_s1 D01_s2'
        self.assertEqual(explainer.variable_data, encoded)
        self.assertEqual([x[0] for x in explainer.explanations['all1'].as_list()], re.split(' ', encoded))