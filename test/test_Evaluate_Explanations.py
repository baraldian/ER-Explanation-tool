from unittest import TestCase

import pandas as pd
import numpy as np
from Evaluate_Explanations import Evaluate_explanation
from LIME_ER_Wrapper import LIME_ER_Wrapper

class Test(TestCase):
    def setUp(self) -> None:
        self.fake_pred = lambda x: np.ones((x.shape[0],)) * 0.5
        def randomf(x):
            return np.random.rand(x.shape[0])
        self.random_pred = randomf

    def test_evaluate_impacts(self):
        lstring1, lstring2, rstring1, rstring2 = 'l1 l2 l3 l4', 'm1 m2 m3 m4', 'r1 r2 r3', 's1 s2 s3 s4 s5'
        left_string = lstring1 + ' ' + lstring2
        right_string = rstring1 + ' ' + rstring2
        el = pd.DataFrame([[1, lstring1, lstring2, rstring1, rstring2]],
                          columns=['id', 'left_A', 'left_B', 'right_A', 'right_B'])


        explainer = LIME_ER_Wrapper(self.random_pred, el, exclude_attrs=[], lprefix='left_',
                                    rprefix='right_', split_expression=r' ')
        num_sample = 5
        impacts_match = explainer.explain(el, num_samples=num_sample)
        ev = Evaluate_explanation(explainer.left_cols, explainer.right_cols,
                                         self.random_pred)
        results = pd.DataFrame(ev.evaluate_impacts(el, impacts_match))
        self.assertEqual(results.id.unique(), [1])

