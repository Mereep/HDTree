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
from hd_tree_classes.split_rule import *
from hd_tree_classes.information_measure import *
from hd_tree_classes.hdtree import HDTreeClassifier
import pandas as pd
import numpy as np
import typing

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
                self.assertFalse(level == 0 and isinstance(rule, TenQuantileSplit), "Tenquantile split"
                                                                                    " should not be first level")
                self.assertFalse(level > 0 and isinstance(rule, TwentyQuantileSplit), "Twenty Quantile split "
                                                                                      "should be only in first level" )

    def _get_data(self) -> typing.Tuple[np.ndarray, np.ndarray, typing.List[str]]:
        df_primitives = pd.read_csv('data/test_data.csv')
        df_primitives.fillna(0, inplace=True)
        primitives = pd.get_dummies(df_primitives.loc[:, 'primitive_type'])
        df_primitives = df_primitives.join(primitives)
        df_primitives = df_primitives.drop('primitive_type', axis=1)
        y_primitives = df_primitives.loc[:, 'target'].values
        df_primitives = df_primitives.drop('target', axis=1)
        x_primitives = df_primitives.iloc[:, 1:-1].values

        return x_primitives, y_primitives, df_primitives.columns[1:-1]
