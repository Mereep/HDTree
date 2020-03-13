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
# @created: 12.08.2019
"""
import typing
from typing import Optional
import numpy as np
from .split_rule import AbstractSplitRule


class Node:
    """
    Represents a node of the HDTree holding the data indices 
    and possibly a split rule
    """

    def __init__(self, assigned_data_indices: typing.List[int],
                 tree: 'AbstractHDTree',
                 parent: 'Node'):
        """
        :param _tree: Decision Tree that this Node belongs to
        :param assigned_data_indices: list of data indices hold by that Node
        """
        assert len(assigned_data_indices) >= 1, "A Node cannot exist when not having at least " \
                                                "one data member (Code: 4723894)"

        if not isinstance(assigned_data_indices, np.ndarray):
            self._assigned_data_indices: np.ndarray = np.array(assigned_data_indices)
        else:
            self._assigned_data_indices: np.ndarray = assigned_data_indices

        self._split_rule = None
        self._parent = parent
        self._tree = tree

    def get_parent(self) -> Optional['Node']:
        return self._parent

    def set_split_rule(self, rule: Optional[AbstractSplitRule]):
        """
        Sets the nodes split rule 
        """
        self._split_rule = rule

    def get_split_rule(self) -> Optional[AbstractSplitRule]:
        """
        Returns the set split rule if any
        """
        return self._split_rule

    def get_tree(self) -> 'AbstractHDTree':
        """
        Returns the tree that node belongs to
        """
        return self._tree

    def set_tree(self, tree: 'AbstractHDTree'):
        self._tree = tree

    def get_score(self) -> Optional[float]:
        """
        Returns the score if the node is split using some rule
        :return:
        """
        return self._tree.get_information_measure().calculate_for_single_node(node=self)

    def get_data_indices(self) -> np.ndarray:
        """
        Returns the indices of the data that are inside that node
        """
        return self._assigned_data_indices

    def get_data(self) -> np.ndarray:
        """
        Returns the samples inside that node
        """
        return self.get_tree().get_train_data()[self.get_data_indices()]

    def get_sample_count(self) -> int:
        """
        Returns amount of samples associated to that node
        :return:
        """
        return len(self.get_data_indices())

    def get_labels(self) -> np.ndarray:
        return self.get_tree().get_train_labels()[self.get_data_indices()]

    def get_children(self) -> Optional[typing.List['Node']]:
        """
        Gets the children of that node, if any
        """
        if self._split_rule and self._split_rule.get_child_nodes():
            return self._split_rule.get_child_nodes()
        else:
            return None

    def make_leaf(self):
        """
        Effectively removes the split rule and children
        :return:
        """
        self.set_split_rule(rule=None)

    def __str__(self):
        children = "no children" if self.get_children() is None else f"{len(self.get_children())} children"
        split_description = "no split rule" if self.get_split_rule() is None else str(self.get_split_rule())
        score = f"(Split Score: {round(self.get_split_rule().get_score(), 3)})" if not self.is_leaf() \
            else  f"(Node Score: {round(self.get_tree().get_information_measure().calculate_for_single_node(node=self), 3)})"
        return f"Node having {len(self.get_labels())} samples and {children} with split rule " \
               f"\"{split_description}\" {score}"

    def is_leaf(self) -> bool:
        """
        Checks if no leafs are attached
        :return:
        """
        return self.get_children() is None # or len(self.get_children()) == 0

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> Optional[typing.List['Node']]:
        """
        Will return the child nodes for the sample using attached split rule
        :param sample:
        :return:
        """
        if not self.is_leaf():
            return self.get_split_rule().get_child_nodes_for_sample(sample=sample)

        return None

    def get_edge_labels(self) -> typing.List[str]:
        """
        Returns the labels of the edges dfined by its split rule if any
        :return:
        """
        if self.is_leaf():
            return []
        else:
            return self.get_split_rule().get_edge_labels()

    def explain_split(self, sample: np.ndarray) -> str:
        """
        Explains reason for that specific split
        :param sample:
        :return:
        """
        if self.is_leaf():
            return "Leaf."

        return self.get_split_rule().explain_split(sample=sample)

    def create_child_node(self, assigned_data_indices: typing.List[int]) -> 'Node':
        """
        Creates a child node having this node as parent
        :param assigned_data_indices:
        :return:
        """
        return self.__class__(assigned_data_indices=assigned_data_indices,
                                      tree=self.get_tree(),
                                      parent=self)
