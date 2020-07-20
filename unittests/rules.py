"""
# Copyright 2018 Professorship Media Informatics, University of Applied Sciences Mittweida
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Richard Vogel, 
# @email: richard.vogel@hs-mittweida.de
# @created: 03.10.2019
"""
import unittest
from hdtree.src.split_rule import *
from hdtree.src.information_measure import *
from hdtree.src.hdtree import HDTreeClassifier
import pandas as pd
import numpy as np
import typing
import os
from os.path import join as pjoin


class RulesTester(unittest.TestCase):

    def test_whitelist(self):
        x, y, cols = self._get_data()

        wl_attributes = ['number_of_triangles',
                         'disturbed_areas_cnt',
                         'yAL_cnt',
                         'yBL_cnt']

        params = dict(allowed_splits=[
            # SingleCategorySplit,
            SmallerThanSplit.build_with_restrictions(
                whitelist_attribute_indices=wl_attributes,
                ),
            TenQuantileAdditiveSplit.build_with_restrictions(
                whitelist_attribute_indices=wl_attributes
            ),
            ],
            information_measure=EntropyMeasure(),
            max_levels=5,
            min_samples_at_leaf=None,
            verbose=False,
            attribute_names=cols
        )

        tree = HDTreeClassifier(**params)
        tree.fit(x, y)

        for i in range(100):
            nodes = tree.extract_node_chain_for_sample(sample=x[np.random.randint(0, len(x))])
            for node in nodes:
                if node.is_leaf():
                    continue

                rule = node.get_split_rule()
                for used_index in  rule.get_state()['split_attribute_indices']:
                    if isinstance(wl_attributes[0], str):
                        used_index = rule.get_tree().get_attribute_names()[used_index]

                    self.assertTrue(used_index in wl_attributes, f"{used_index} is not in whitelist")

    def test_blacklist(self):
        x, y, cols = self._get_data()

        bl_attributes = [
            'number_of_triangles',
            'disturbed_areas_cnt',
            'yAL_cnt',
            'yBL_cnt']

        params = dict(allowed_splits=[
            # SingleCategorySplit,
            SmallerThanSplit.build_with_restrictions(
                blacklist_attribute_indices=bl_attributes,
                ),
            TenQuantileAdditiveSplit.build_with_restrictions(
                blacklist_attribute_indices=bl_attributes
                ),
            ],
            information_measure=EntropyMeasure(),
            max_levels=5,
            min_samples_at_leaf=None,
            verbose=False,
            attribute_names=cols
        )

        tree = HDTreeClassifier(**params)
        tree.fit(x, y)

        for i in range(100):
            nodes = tree.extract_node_chain_for_sample(sample=x[np.random.randint(0, len(x))])
            for node in nodes:
                if node.is_leaf():
                    continue

                rule = node.get_split_rule()
                for used_index in  rule.get_state()['split_attribute_indices']:
                    if isinstance(bl_attributes[0], str):
                        used_index = rule.get_tree().get_attribute_names()[used_index]

                    self.assertFalse(used_index in bl_attributes, f"{used_index} is in blacklist")

    def test_min_max_levels(self):
        x, y, cols = self._get_data()

        params = dict(allowed_splits=[
            # SingleCategorySplit,
            TwentyQuantileSplit.build_with_restrictions(
                min_level=0,
                max_level=0
            ),
            TenQuantileSplit.build_with_restrictions(
                min_level=1,
            ),
        ],
            information_measure=EntropyMeasure(),
            max_levels=3,
            min_samples_at_leaf=None,
            verbose=False,
            attribute_names=cols
        )

        tree = HDTreeClassifier(**params)
        tree.fit(x, y)

        for i in range(100):
            nodes = tree.extract_node_chain_for_sample(sample=x[np.random.randint(0, len(x))])
            for level, node in enumerate(nodes):
                if node.is_leaf():
                    continue

                rule = node.get_split_rule()
                self.assertFalse(level == 0 and isinstance(rule, TenQuantileSplit), "Ten Quantile split"
                                                                                    " should not be first level")
                self.assertFalse(level > 0 and isinstance(rule, TwentyQuantileSplit), "Twenty Quantile split "
                                                                                      "should be only in first level" )

    def test_merge_quantile_split(self):
        split1 = TwentyQuantileSplit(node=None)
        split1._state = {
            'split_value': 1.,
            'quantile': 2,
            'split_attribute_indices': [1]
        }
        split2 = TenQuantileSplit(node=None)
        split2._state = {
            'split_value': 0.5,
            'quantile': 1,
            'split_attribute_indices': [1]
        }

        # check addition of two Quantile Splits


        # ... direction 1
        split_combine = split1 + split2
        self.assertIsInstance(split_combine, TwentyQuantileSplit, "Split should downcast to twenty quantile")
        self.assertEqual(split_combine.get_state()['split_value'], 0.5, "Value should be the lower of both")

        # ... direction 2 (Kommunikativ)
        split_combine = split2 + split1
        self.assertIsInstance(split_combine, TwentyQuantileSplit, "Split should downcast to twenty quantile")
        self.assertEqual(split_combine.get_state()['split_value'], 0.5, "Value should be the lower of both")

        # non-working split
        split_random = SingleCategorySplit(None)
        split_random.set_state({'split_attribute_indices': [1]})

        split_combine = split2 + split_random
        self.assertEqual(split_combine, (split2, split_random))

    def test_merge_close_to_median_split(self):

        # test split 2 inside
        split1 = CloseToMedianSplit(node=None)
        split1.set_state({'median': 2, 'stdev': 1, 'split_attribute_indices': [1]})

        split2 = CloseToMedianSplit(node=None)
        split2.set_state({'median': 2.1, 'stdev': 0.2, 'split_attribute_indices': [1]})

        new_split = split1 + split2

        self.assertIsInstance(new_split, CloseToMedianSplit, "Should produce exactly one Median Split")
        self.assertEqual(2.1, new_split.get_state()['median'], "Should be split 2s Median")

        # test split 2 hanging out right
        split2.set_state({'median': 2.1, 'stdev': 1, 'split_attribute_indices': [1]})
        new_split = split1 + split2
        expected_median = (2.1+2) / 2
        expected_stdev = 2 * (expected_median - (split2.get_state()['median'] - 0.5 * split2.get_state()['stdev']))

        self.assertIsInstance(new_split, CloseToMedianSplit, "Should produce exactly one Median Split")
        self.assertEqual(expected_median, new_split.get_state()['median'], "Median should adapt to new range")
        self.assertAlmostEqual(expected_stdev, new_split.get_state()['stdev'], 5, "Standard Dev should "
                                                                                  "adapt to new range")

        # test split 2 hanging out left
        split2.set_state({'median': 1.9, 'stdev': 1, 'split_attribute_indices': [1]})
        new_split = split1 + split2
        expected_median = (1.9 + 2.) / 2
        expected_stdev = 2 * (expected_median - (split1.get_state()['median'] - 0.5 * split1.get_state()['stdev']))

        self.assertIsInstance(new_split, CloseToMedianSplit, "Should produce exactly one Median Split")
        self.assertEqual(expected_median, new_split.get_state()['median'], "Median should adapt to new range")
        self.assertAlmostEqual(expected_stdev, new_split.get_state()['stdev'], 5, "Standard Dev should "
                                                                                  "adapt to new range")

        # ... eating quantile split (im inside interval)
        split3 = TenQuantileSplit(node=None)
        split3.set_state({'split_value': 2})
        new_split = split1 + split2

        self.assertIsInstance(new_split, CloseToMedianSplit, "Close to Median Split should reduce Quantile split")

        # ... merging quantile split (I hang out right of interval)
        split4 = TenQuantileSplit(node=None)
        split4.set_state({'split_value': 2, 'split_attribute_indices': [1, 2]})
        new_split = split1 + split4
        expected_median = ((split1.get_state()['median'] - split1.get_state()['stdev'] * 0.5) + 2) / 2
        expected_stdev = (2 - expected_median)
        self.assertIsInstance(new_split, CloseToMedianSplit, "Close to Median Split should reduce Quantile split")
        self.assertEqual(expected_median, new_split.get_state()['median'], "Close to Median Split "
                                                                           "should be 2")
        self.assertEqual(expected_stdev, new_split.get_state()['stdev'], "stdev should adapt correctly")

    def test_merge_fixed_split(self):

        # eat quantile split
        split1 = TwentyQuantileSplit(node=None)
        split1._state = {
            'split_value': 1.,
            'quantile': 2,
            'split_attribute_indices': [1]
        }

        split2 = FixedValueSplit(node=None)
        split2._state = {
            'value': 0.3,
            'split_attribute_indices': [1]
        }

        split_merge = split1 + split2

        self.assertIsInstance(split_merge, FixedValueSplit, "Split should downcast to a fixed split")
        self.assertEqual(split_merge.get_state()['value'], 0.3, "Value should be as defined")

        # check edge case being a string
        split3 = FixedValueSplit(node=None)
        split3._state = {
            'value': 'a',
            'split_attribute_indices': [1]
        }
        split_merge = split3 + split1
        self.assertEqual(split_merge, (split3, split1, ), "Should not cast together")

        # eat close to median split
        split4 = CloseToMedianSplit(node=None)
        split4._state = {
            'median': 0.3,
            'stdev': 1,
            'split_attribute_indices': [1]
        }

        new_split = split4 + split2
        self.assertIsInstance(new_split, FixedValueSplit)

    def test_merge_range_split(self):

        # consume RangeSplit
        split1 = TenQuantileRangeSplit(node=None)
        split2 = TenQuantileRangeSplit(node=None)

        split1._state = {'lower_bound': 1,
                         'upper_bound': 2,
                         'split_attribute_indices': [2]}

        split2._state = {'lower_bound': 1.2,
                         'upper_bound': 1.4,
                         'split_attribute_indices': [2]
                         }

        merge = split1 + split2

        self.assertIsInstance(merge, AbstractQuantileRangeSplit)
        self.assertEqual(merge.get_upper_bound(), split2.get_upper_bound())
        self.assertEqual(merge.get_lower_bound(), split2.get_lower_bound())

        # consume CloseToMedianSplit
        split3 = CloseToMedianSplit(node=None)
        split3._state = {'median': 1.5,
                         'stdev': 3,
                         'split_attribute_indices': [2]}

        merge = split1 + split3
        self.assertIsInstance(merge, AbstractQuantileRangeSplit)
        self.assertEqual(merge.get_upper_bound(), split1.get_upper_bound())
        self.assertEqual(merge.get_lower_bound(), split1.get_lower_bound())

        # consume QuantileSplit
        split4 = TenQuantileSplit(node=None)
        split4._state = {'split_value': 3,
                         'split_attribute_indices': [2]}

        merge = split1 + split4
        self.assertIsInstance(merge, AbstractQuantileRangeSplit)
        self.assertEqual(merge.get_upper_bound(), split1.get_upper_bound())
        self.assertEqual(merge.get_lower_bound(), split1.get_lower_bound())

    def test_simplify_rules(self):
        split1 = TwentyQuantileSplit(node=None)
        split1._state = {
            'split_value': 1.,
            'quantile': 2,
            'split_attribute_indices': [1]
        }

        split2 = FixedValueSplit(node=None)
        split2._state = {
            'value': 0.3,
            'split_attribute_indices': [1]
        }

        split3 = SingleCategorySplit(None)
        split3.set_state({'split_attribute_indices': [4]})

        split4 = TwentyQuantileSplit(node=None)
        split4._state = {
            'split_value': 1.,
            'quantile': 2,
            'split_attribute_indices': [2]
        }

        split5 = TwentyQuantileSplit(node=None)
        split5._state = {
            'split_value': 0.,
            'quantile': 2,
            'split_attribute_indices': [2]
        }

        splits = [split1, split2, split3, split4, split5]

        rules_simple = simplify_rules(splits)

        self.assertEqual(3, len(rules_simple), "Two rules should be eaten")

    def test_merge_with_sample(self):
        #### Quantile + Quantile
        split1 = TenQuantileSplit(node=None)
        split2 = TenQuantileSplit(node=None)

        split1._state = {'split_value': 1,
                         'split_attribute_indices': [2]}

        split2._state = {'split_value': 2,
                         'split_attribute_indices': [2]
                         }

        rules = [split1, split2]
        # case I: Below
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 1.-1e-10]))
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit, "is at lower bound --> should be merged "
                                                           "to quantile split again")
        # case II: Between
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 1.5]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "is inside range -> should be MergedQuantileSplit")
        self.assertEqual(rule.get_upper_bound(), 2)
        self.assertEqual(rule.get_lower_bound(), 1)

        # case II: Above
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 2.]))
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit, "is at upper bound --> should be merged to "
                                                           "quantile split again")
        self.assertEqual(rule.get_split_value(), 2)

        ### QuantileRange + QuantileRange
        higher = TenQuantileRangeSplit(node=None)
        lower = TenQuantileRangeSplit(node=None)

        higher._state = {'upper_bound': 6,
                         'lower_bound': 3,
                         'split_attribute_indices': [2]}

        lower._state = {'upper_bound': 4,
                        'lower_bound': 0,
                        'split_attribute_indices': [2]}

        rules = [lower, higher]
        # test case 1: Sample in both
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 3.5]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "Sample is within both -> one range split")
        self.assertEqual(rule.get_lower_bound(), higher.get_lower_bound(), "New split should start above higher split")
        self.assertEqual(rule.get_upper_bound(), lower.get_upper_bound(), "New split should end below lower split")

        # test case 2: Sample in lower only
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 2]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "Sample should be only in lower range")
        self.assertEqual(rule.get_lower_bound(), lower.get_lower_bound(), "New split should start at loweesz pos")
        self.assertEqual(rule.get_upper_bound(), higher.get_lower_bound(), "New split ")

        # test case V: Sample in upper only
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 5.]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "Sample only in higher range")
        self.assertEqual(rule.get_lower_bound(), lower.get_upper_bound(), "Should be the lowest thing after the "
                                                                          "thing we know its NOT inside")
        self.assertEqual(rule.get_upper_bound(), higher.get_upper_bound(), "highest thing")

        # test case 3: Sample above all
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 7]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit, "Sample is above all -> Quantile Split")
        self.assertEqual(rule.get_split_value(), higher.get_upper_bound(), "Split should separate on highest spot")

        # test case 4: Sample below all
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., -1.]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit, "Sample is below all -> Quantile Split")
        self.assertEqual(rule.get_split_value(), lower.get_lower_bound(), "Sample smaller than smallest")

        # test case 6: No rule overalp and sample between
        higher._state = {'upper_bound': 1,
                         'lower_bound': 0,
                         'split_attribute_indices': [2]}

        lower._state = {'upper_bound': -1,
                        'lower_bound': -2,
                        'split_attribute_indices': [2]}

        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., -0.5]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "Sample between two ranges -> RangeSplit")
        self.assertEqual(rule.get_lower_bound(), lower.get_upper_bound(), "Should start at end of lower rule")
        self.assertEqual(rule.get_upper_bound(), higher.get_lower_bound(), "Should end at start of upper rule")

        ### QuantileRange + Range

        # ..Class I: Overlap
        split1 = TenQuantileSplit(node=None)
        split1._state = {'split_value': 5,
                         'split_attribute_indices': [2]}

        split2 = TenQuantileRangeSplit(node=None)

        split2._state = {'upper_bound': 6,
                         'lower_bound': 3,
                         'split_attribute_indices': [2]}

        # ... I.I below
        rules = [split1, split2]

        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 0]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit, "Sample is below all -> Quantile Split")
        self.assertEqual(rule.get_split_value(), split2.get_lower_bound(), "Sample smaller than smallest")

        # ... I.II in split below quantile split
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 4]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit, "Sample in range")
        self.assertEqual(rule.get_lower_bound(), split2.get_lower_bound())
        self.assertEqual(rule.get_upper_bound(), split1.get_split_value())

        # .... I.III above quantile and inside range
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 5.5]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit)
        self.assertEqual(rule.get_lower_bound(), split1.get_split_value())
        self.assertEqual(rule.get_upper_bound(), split2.get_upper_bound())

        # .... I.IV above all
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 7.2]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit)
        self.assertEqual(rule.get_split_value(), split2.get_upper_bound())

        # Case II: Quantile split BELOW RangeSplit
        split1 = TenQuantileSplit(node=None)
        split1._state = {'split_value': 1,
                         'split_attribute_indices': [2]}

        split2 = TenQuantileRangeSplit(node=None)

        split2._state = {'upper_bound': 6,
                         'lower_bound': 3,
                         'split_attribute_indices': [2]}

        rules = [split1, split2]

        #  II.I sample below quantile split
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 0]))
        self.assertEqual(len(new_rules), 1)
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit)
        self.assertEqual(rule.get_split_value(), split1.get_split_value())

        # II.II sample between both
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 2]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit)
        self.assertEqual(rule.get_lower_bound(), split1.get_split_value())
        self.assertEqual(rule.get_upper_bound(), split2.get_lower_bound())

        #  II.III in range
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 4]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit)
        self.assertEqual(rule.get_lower_bound(), split2.get_lower_bound())
        self.assertEqual(rule.get_upper_bound(), split2.get_upper_bound())


        #  II.IV above all
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 100]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit)
        self.assertEqual(rule.get_split_value(), split2.get_upper_bound())

        # Case III quantile split above
        split1 = TenQuantileSplit(node=None)
        split1._state = {'split_value': 8,
                         'split_attribute_indices': [2]}

        split2 = TenQuantileRangeSplit(node=None)

        split2._state = {'upper_bound': 6,
                         'lower_bound': 3,
                         'split_attribute_indices': [2]}

        rules = [split1, split2]

        #  III.I sample below quantile split
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 2]))
        self.assertEqual(len(new_rules), 1)
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit)
        self.assertEqual(rule.get_split_value(), split2.get_lower_bound())

        #  II.II in range
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 4]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit)
        self.assertEqual(rule.get_lower_bound(), split2.get_lower_bound())
        self.assertEqual(rule.get_upper_bound(), split2.get_upper_bound())

        #  II.III above range below quantile
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 7]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, MergedRangeSplit)
        self.assertEqual(rule.get_lower_bound(), split2.get_upper_bound())
        self.assertEqual(rule.get_upper_bound(), split1.get_split_value())

        #  II.IV all
        new_rules = simplify_rules(rules=rules, sample=np.array([0., 0., 9]))
        self.assertEqual(len(new_rules), 1, "Rules should merge")
        rule = new_rules[0]
        self.assertIsInstance(rule, AbstractQuantileSplit)
        self.assertEqual(rule.get_split_value(), split1.get_split_value())


    def _get_data(self) -> typing.Tuple[np.ndarray, np.ndarray, typing.List[str]]:
        df_primitives = pd.read_csv(pjoin(os.path.dirname(os.path.abspath(__file__)),
                                          'data/test_data.csv'))
        df_primitives.fillna(0, inplace=True)
        primitives = pd.get_dummies(df_primitives.loc[:, 'primitive_type'])
        df_primitives = df_primitives.join(primitives)
        df_primitives = df_primitives.drop('primitive_type', axis=1)
        y_primitives = df_primitives.loc[:, 'target'].values
        df_primitives = df_primitives.drop('target', axis=1)
        x_primitives = df_primitives.iloc[:, 1:-1].values

        return x_primitives, y_primitives, df_primitives.columns[1:-1]

    def test_fixed_chain_rule(self):
        X, y, cols = self._get_data()

        tree_params = dict(allowed_splits=[TenQuantileSplit],
                           max_levels=3,
                           min_samples_at_leaf=10, attribute_names=cols,
                           information_measure=EntropyMeasure())

        tree_1 = HDTreeClassifier(**tree_params)
        tree_1.fit(X, y)

        node_1 = tree_1.get_node_for_tree_walk([1, 0, 1])
        rule_new_1 = FixedChainRule.from_node(node_1, name="Tester")

        tree_params_2 = dict(allowed_splits=[rule_new_1],
                       max_levels=3,
                       attribute_names=cols,
                       min_samples_at_leaf=10, information_measure=EntropyMeasure())

        tree_2 = HDTreeClassifier(**tree_params_2)
        tree_2.fit(X, y)
        graph = tree_2.generate_dot_graph()

    def test_handpicked_cases(self):
        """
        These are just some cases that went wrong while testing
        :return:
        """

        # casscade of 3 -> 1
        data_1, data_2 = np.array([73]),  np.array([55, 73])

        split_3 = TenQuantileSplit(node=None)
        split_3._state = {
            'split_value': 68,
            'quantile': 1,
            'split_attribute_indices': [0]
        }

        split_1 = TenQuantileSplit(node=None)
        split_1._state = {
            'split_value': 75,
            'quantile': 1,
            'split_attribute_indices': [1]
        }

        split_2 = TenQuantileSplit(node=None)
        split_2._state = {
            'split_value': 67,
            'quantile': 1,
            'split_attribute_indices': [1]
        }

        class TreeDummy:
            def __init__(self, names):
                self.names = names

            def get_attribute_names(self):
                return self.names

        class NodeDummy:
            def __init__(self, names):
                self.names = names
                self.tree = TreeDummy(names)

            def get_tree(self):
                return self.tree

        node_dummy_1 = NodeDummy(['1'])
        node_dummy_2 = NodeDummy(['0', '1'])

        split_3.get_tree = lambda: node_dummy_1.get_tree()
        split_1.get_tree = lambda: node_dummy_2.get_tree()
        split_2.get_tree = lambda: node_dummy_2.get_tree()

        split_3.get_node = lambda: node_dummy_1
        split_1.get_node = lambda: node_dummy_2
        split_2.get_node = lambda: node_dummy_2

        lookup = {
            node_dummy_1.get_tree(): data_1,
            node_dummy_2.get_tree(): data_2
        }

        res = simplify_rules([split_1, split_2, split_3], model_to_sample=lookup)

        self.assertEqual(len(res), 1, "Should melt down to one rule")
        self.assertIsInstance(res[0], MergedRangeSplit, "67 split should be eaten completely and the "
                                                                  "other two should merge to appropriate range")
        self.assertEqual(res[0].get_state()['lower_bound'], 68)
        self.assertEqual(res[0].get_state()['upper_bound'], 75)

        # merge fixed value split with quantile split
        split_2 = TenQuantileSplit(node=None)
        split_2._state = {
            'split_value': 67,
            'quantile': 1,
            'split_attribute_indices': [1]
        }

        split_1 = TenQuantileSplit(node=None)
        split_1._state = {
            'split_value': 69,
            'quantile': 1,
            'split_attribute_indices': [0]
        }

        split_2 = FixedValueSplit(node=None)
        split_2._state = {
            'value': 68,
            'split_attribute_indices': [0]
        }

        res = simplify_rules([split_1, split_2], sample=np.array([67]))
        self.assertEqual(len(res), 2)

        res = simplify_rules([split_1, split_2], sample=np.array([68]))
        self.assertEqual(len(res), 1)


