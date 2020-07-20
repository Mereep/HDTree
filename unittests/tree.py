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
# @created: 18.11.2019
"""
import typing
import numpy as np
import pandas as pd
from os.path import join as pjoin
import unittest
import os
from hdtree import *
from hdtree.src.hdtree import AbstractHDTree


class RulesTester(unittest.TestCase):

    def test_simplification(self):
        x, y, cols = self._get_data()

        params = dict(allowed_splits=[
            TenQuantileSplit
        ],
            information_measure=EntropyMeasure(),
            max_levels=5,
            min_samples_at_leaf=None,
            verbose=False,
            attribute_names=cols
        )

        tree = HDTreeClassifier(**params)
        tree.fit(x, y)
        pred_cumbersome = list(tree.predict(x))
        nodes = tree.get_all_nodes()
        tree_simple = tree.simplify(return_copy=True)
        nodes_simple = tree_simple.get_all_nodes()
        pred_simple = list(tree_simple.predict(x))

        self.assertLess(len(nodes_simple), len(nodes), "Rules should be less than before")
        self.assertListEqual(pred_simple, pred_cumbersome)

    def test_feature_importance(self):
        x, y, cols = self._get_data()

        wl_attributes = [0, 2, 3, 1]

        params = dict(allowed_splits=[
            # SingleCategorySplit,
            TenQuantileSplit.build_with_restrictions(
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

        importances = tree.compute_feature_importances()

        self.assertEqual(np.sum(importances), 1., "Sum of importances has to be exactly one by definition")
        self.assertTrue(sum(np.where(importances > 0, 1, 0)) <= len(wl_attributes), "Attributes that are not "
                                                                                    " whitelisted cannot contribute " 
                                                                                    "to"
                                                                                    "feature importance")

        self.assertEqual(sum(importances[4:]), 0, "We only whitelisted attributes up to index 3, hence"
                                                  " later attributes should not have feature importances")

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

    def test_copy_tree(self):
        x, y, cols = self._get_data()

        wl_attributes = [0, 2, 3, 1]

        params = dict(allowed_splits=[
            # SingleCategorySplit,
            TenQuantileSplit.build_with_restrictions(
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

        cpy = tree.__copy__()

        self.assertIsInstance(cpy, AbstractHDTree, "Should return a valid HDTree")
        self.assertIs(tree.get_train_data(), cpy.get_train_data(), "Data should be referenced over")
        self.assertIs(tree.get_train_labels(), cpy.get_train_labels(), "Labels should be referenced over")

        nodes_tree = tree.get_all_nodes_below_node(node=None)
        nodes_cpy = cpy.get_all_nodes_below_node(node=None)

        self.assertEqual(len(nodes_tree), len(nodes_cpy), "There should be the same amount of nodes")

        self.assertSequenceEqual([node.get_split_rule().user_readable_description() for node in nodes_tree if node.get_split_rule() is not None],
                                 [node.get_split_rule().user_readable_description() for node in nodes_cpy if node.get_split_rule() is not None],
                                 'Nodes should expand in same order and be the same things'
                                 )

        self.assertTrue(all([node.get_tree() is cpy for node in nodes_cpy]), "node copies should reference to tree copy")
        self.assertTrue(all([node.get_tree() is tree for node in nodes_tree]), "node originals should reference to original tree")
        self.assertIs(nodes_cpy[-1].get_parent().get_parent().get_tree(), cpy, 'Parents should link to correct tree')
        self.assertIs(nodes_tree[-1].get_parent().get_parent().get_tree(), tree, 'Parents should link to correct tree')

