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
from typing import Optional
from abc import ABC, abstractmethod
from .node import Node
import numpy as np
from collections import Counter


class AbstractInformationMeasure(ABC):
    """
    Represents a real valued function [0..1]
    that represents the value of the nodes

    call the instance for calculation
    """

    @abstractmethod
    def calculate_for_children(self, parent_node: Node) -> Optional[float]:
        """
        Calculate pureness of the node using (if possible)
        """
        pass

    def calculate_for_single_node(self, node: Node, normalize: bool= True) -> Optional[float]:
        """
        calculates score of a single node
        this is just a wrapper around @see calculate_for_labels
        :param node:
        :param normalize: Value will be between 0..1
        :return:
        """
        labels = node.get_labels()
        return self.calculate_for_labels(labels=labels, normalize=normalize)

    @abstractmethod
    def supports_regression(self) -> bool:
        pass

    @abstractmethod
    def supports_classification(self) -> bool:
        pass

    @abstractmethod
    def calculate_for_labels(self, labels: np.ndarray, normalize: bool=True) -> float:
        pass

    def __call__(self, parent_node: Node) -> Optional[float]:
        val = self.calculate_for_children(parent_node=parent_node)
        if val is None:
            return val

        assert val > -1e-5 and val < 1. + 1e-5, f"_calculate_for_nodes has to return a value between [0..1] but " \
                                                f"returned {val} (Code: 33273928)"

        return val


class RelativeAccuracyMeasure(AbstractInformationMeasure):
    """
    Counts relative pureness of the nodes
    by 1/node_proportion * (accuracy_child)
    """

    def supports_regression(self):
        return False

    def supports_classification(self):
        return True

    def calculate_for_children(self, parent_node: Node):
        child_nodes = parent_node.get_children()
        if child_nodes is None:
            child_nodes = []

        node: Optional[Node] = None
        # calculate accuracy for each node

        n_total_samples = sum(len(node.get_labels()) for node in child_nodes)
        pureness_childs = 0.
        for node in child_nodes:
            child_cats = node.get_labels()
            accuracy = self.calculate_for_single_node(node=node)
            pureness_childs += accuracy * len(child_cats)/n_total_samples

        return pureness_childs

    def calculate_for_labels(self, labels: np.ndarray, normalize: bool=True):
        node_cats = labels
        most_common_class = Counter(node_cats).most_common()[0][0]
        accuracy = sum(node_cats == most_common_class) / len(node_cats)

        return (accuracy - 0.5) * 2.


class EntropyMeasure(AbstractInformationMeasure):
    """
    Counts relative pureness of the nodes
    by 1/node_proportion * (entropy child)
    """

    def supports_regression(self):
        return False

    def supports_classification(self):
        return True

    def calculate_for_children(self, parent_node: Node):
        child_nodes = parent_node.get_children()
        if child_nodes is None:
            child_nodes = []

        # calculate accuracy for each node
        n_total_samples = sum(len(node.get_labels()) for node in child_nodes)
        pureness_childs = 0.

        # print("Child node ids", [id(cn) for cn in child_nodes], "ladsldasö33")
        for node in child_nodes:
            child_cats = node.get_labels()
            pureness = self.calculate_for_single_node(node=node)
            # print(pureness, node)
            pureness_childs += pureness * (len(child_cats) / n_total_samples)

        # print(pureness_childs, "Total")
        return pureness_childs

    def calculate_for_labels(self, labels: np.ndarray, normalize=True):
        node_cats = labels
        n_cats = len(set(node_cats))

        # maximum clean if only one category
        if n_cats == 1:
            return 1.

        counts = Counter(node_cats)
        val = 0
        n_samples = len(node_cats)
        for cls, amount in counts.items():
            p = amount / n_samples
            part = p * np.log2(p)

            # if normalize:
            #     part /= np.log2(3)

            val += part

        #return 1 - val

        if normalize:
            # normalize 0 .. 1
            #p_worst = (n_samples / n_cats) / n_samples
            #max_entropy = (p_worst * np.log2(p_worst)) * n_cats

            max_entropy = (0.5 * np.log2(0.5)) * n_cats
            val /= max_entropy


        return 1. - val


class GiniMeasure(AbstractInformationMeasure):
    """ Common Gini Score
    as defined by sum_{p_i}(p_i^2)
    """

    def supports_regression(self):
        return False

    def supports_classification(self):
        return True

    def calculate_for_children(self, parent_node: Node):
        child_nodes = parent_node.get_children()
        if child_nodes is None:
            child_nodes = []

        node: Optional[Node] = None
        # calculate accuracy for each node

        n_total_samples = sum(len(node.get_labels()) for node in child_nodes)
        pureness_childs = 0.
        for node in child_nodes:
            child_cats = node.get_labels()
            accuracy = self.calculate_for_single_node(node=node)
            pureness_childs += accuracy * len(child_cats)/n_total_samples

        return pureness_childs

    def calculate_for_labels(self, labels: np.ndarray, normalize: bool=True):
        n_labels = len(labels)
        counts = Counter(labels)

        p = 0
        for label, cnt in counts.items():
            p += (cnt / n_labels) ** 2

        return p
