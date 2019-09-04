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
from abc import ABC, abstractmethod
from .node import Node
import numpy as np
from .information_measure import AbstractInformationMeasure
from .split_rule import AbstractSplitRule, LessThanHalfOfSplit
import logging
from collections import Counter


class AbstractHDTree(ABC):
    def __init__(self,
                 allowed_splits: typing.List[typing.Type[AbstractSplitRule]],
                 information_measure: AbstractInformationMeasure,
                 attribute_names: typing.List[str] = None,
                 max_levels: Optional[int] = None,
                 min_samples_at_leaf: Optional[int] = None,
                 verbose: bool = False):
        """
        Will initialize the model with preparing state
        """
        self._train_data = None
        self._labels = None
        self._node_head = None
        self._min_samples_at_leaf = min_samples_at_leaf
        self._max_levels = max_levels
        self._allowed_splits = allowed_splits
        self._information_measure = information_measure
        self._verbose = verbose
        self._attribute_types: typing.List[str] = None
        self._attribute_names = attribute_names
        self._is_fit = False

    def is_fit(self) -> bool:
        return self._is_fit

    def get_attribute_names(self) -> typing.List[str]:
        return self._attribute_names

    def _output_message(self, message: str, only_if_verbose=True):
        """
        Just logs a message
        """
        if (only_if_verbose and self._verbose) or not only_if_verbose:
            logging.getLogger(self.__class__.__name__).info(message)
            print(message)  # debug

    def get_train_data(self) -> Optional[np.ndarray]:
        """
        Will return the initially handed over train data
        """
        return self._train_data

    def get_train_labels(self) -> Optional[np.ndarray]:
        """
        Will return the initially handed over labels
        """
        return self._labels

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Predicts samples in format [n_samples, n_features]
        :param samples:
        :return:
        """
        assert self.is_fit(), "Tree is fit on data, hence cannot predict (Code: 2842094823)"
        assert len(samples.shape) == 2, "Data has to be in format n_samples x n_features (Code: 234234234)"
        assert samples.shape[1] == len(self.get_attribute_names()), "Amount of labels has to match amount of " \
                                                                 "features (Code: 23842039482)"

        node = self._node_head
        ret_classes = []

        for sample in samples:
            ret_classes.append(self._predict_sample(sample=sample))

        return np.array(ret_classes)

    @abstractmethod
    def _predict_sample(self, sample: np.ndarray):
        """
        Predicts exactly one sample
        :param sample:
        :return:
        """
        pass

    def explain_decision(self, sample: np.ndarray) -> np.str:
        """
        Returns a human readable decision
        :return:
        """
        childs = self._follow_for_sample_to_leafs(start_node=self._node_head, sample=sample)

        # each child contributes to the decision (eventually, atm not)
        sample_dict = {key: val for key, val in zip(self._attribute_names, sample)}
        ret = f"Sample: \n {sample_dict}\n\n"

        ret += f"Predicted sample as \"{self._predict_sample(sample=sample)}\" because of: \n"

        for i, child in enumerate(childs):
            path = [child]
            ret += f"Explanation {i+1}:\n"

            while child.get_parent() is not None:
                path.append(child.get_parent())
                child = child.get_parent()

            path.reverse()

            for step, node in enumerate(path):
                ret += f"Step {step+1}: {node.explain_split(sample=sample)}\n"

            ret += '---------------------------------\n'
        return ret

    def _follow_for_sample_to_leafs(self, start_node: Node, sample: np.ndarray):
        """
        Follows the tree recursively down to the leafs returning all leaf-nodes the sample belongs to
        :param start_node:
        :param sample:
        :return:
        """
        childs = start_node.get_child_nodes_for_sample(sample=sample)
        leafs = []

        if not childs:
            childs = []
            leafs = [start_node]
            self._output_message("Warning: That tree does not have any nodes except head. "
                                 "Predictions will be the datas' priors", only_if_verbose=False)
        for child in childs:
            if child.is_leaf():
                leafs.append(child)
            else:
                childs += self._follow_for_sample_to_leafs(start_node=child, sample=sample)

        return leafs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Will train the Tree
        :raises Exception if preconditions are not met
        """

        self._labels = y
        self._train_data = X

        status, msg = self._check_preconditions()
        if status is False:
            raise Exception(f"Precondition check failed due to: \n\"{msg}\"\n(Code: 347239847239)")

        # create head node over all indices
        self._node_head = self._create_node_for_data_indices(data_indices=[*range(0, len(self.get_train_data()))])

        current_nodes_to_split = [self._node_head]

        level = 0

        while True:
            self._output_message(f"Splitting level {level+1}")
            collected_children = []

            for node_to_split in current_nodes_to_split:
                n_node_samples = len(node_to_split.get_data_indices())

                # check if we want to split that node
                # we do not if we only have one sample or <= requested minimum amount of samples
                #if (self._min_samples_at_leaf is not None and self._min_samples_at_leaf >= n_node_samples) or \
                #  n_node_samples <= 1:
                #    continue

                # do not split if already clean
                if node_to_split.get_score() == 1.:
                    continue

                # split the node
                min_samples = 1 if not self._min_samples_at_leaf else self._min_samples_at_leaf
                self._split_node(node_to_split=node_to_split, min_samples_leaf=min_samples)

                # collect children of node the nodes if any
                children_of_node = node_to_split.get_children()
                if children_of_node:
                    collected_children += children_of_node

            current_nodes_to_split = collected_children
            level += 1

            if self._max_levels is not None and self._max_levels == level:
                break

            # nothing left to split we are done
            if len(current_nodes_to_split) == 0:
                break

        self._is_fit = True

    def _split_node(self, node_to_split: Node, min_samples_leaf: int= 1):
        """
        Does a node split
        :param node_to_split:
        :param min_samples_leaf:
        :return:
        """
        splits = []
        for split_type in self._allowed_splits:
            splitter = split_type(node=node_to_split)
            score = splitter()
            if score is not None:
                # check if each leaf has at least min_members_leaf entries
                # skip if not
                node_too_small = False
                if splitter.get_child_nodes() and len(splitter.get_child_nodes()) > 0:
                    for node in splitter.get_child_nodes():
                        if node.get_sample_count() < min_samples_leaf:
                            node_too_small = True

                    if not node_too_small:
                        splits.append((score, splitter))

        if len(splits) > 0:
            best_split = sorted(splits, key=lambda data: data[0], reverse=True)[0][1]
            node_to_split.set_split_rule(best_split)


    def _create_node_for_data_indices(self, data_indices: typing.List[int]) -> Node:
        """
        Creates a node for given data indices
        """
        node = Node(assigned_data_indices=data_indices, tree=self, parent=None)

        return node

    def _guess_attribute_types(self) -> typing.List[str]:
        """
        Will check the handed over data and return
        the types of attributes in the data
        will return values categorical, numerical, other
        """
        attributes = []
        data = self.get_train_data()
        attribute = "other"
        for i in range(data.shape[1]):
            vals = data[:, i]

            # categorical?
            if np.all([isinstance(val, (str, type(None))) or np.isnan(val) for val in vals]):
                attribute = 'categorical'
            # numeric?
            elif np.all([isinstance(val, (type(None), float, int,  np.nan)) or np.isnan(val) for val in vals]):
                attribute = 'numerical'

            attributes.append(attribute)

        return attributes

    @abstractmethod
    def _check_preconditions(self) -> typing.Tuple[bool, str]:
        """
        Should check if input data is ok
        :returns: Tuple of status (ok / not ok) and message
        """
        data = self.get_train_data()
        labels = self.get_train_labels()

        if not len(labels) == len(data):
            return False, "Amount of labels does not comply with amount of data handed over (Code: 39847298374)"

        if not len(data.shape) == 2:
            return False, "Input data has to be in format (n_samples, n_features) (Code: 83472839472398)"

        if not len(labels.shape) == 1:
            return False, "Labels have to be in format (n_samples) (Code: 23874092374)"

        if not data.shape[0] > 0 or not data.shape[1] > 0:
            return False, "There has to be at least one sample and at least one Attribute (Code: 4723984723894)"

        # create attribute names if not given
        if not self._attribute_names:
            self._attribute_names = [f"Attribute {i + 1}" for i in range(0, data.shape[1])]

        # check if enough attribute names given
        if not len(self._attribute_names) == data.shape[1]:
            return False, "Amount of attribute names does not match amount of attributes in data (Code: 347823894)"

        attribute_types = self._guess_attribute_types()
        if not np.all([attr_type in ["categorical", "numerical"] for attr_type in attribute_types]):
            return False, "Attributes have to be numerical or categorical, however None is allowed as value (Code: 742398472398)"

        self._attribute_types = attribute_types

        return True, ""

    def get_attribute_types(self) -> Optional[typing.List[str]]:
        """
        Will return the guessed attribute types
        Note that we only support categorical (str) and numerical (float)
        """
        return self._attribute_types

    def get_information_measure(self) -> AbstractInformationMeasure:
        """
        Returns the assigned information measure
        """
        return self._information_measure

    @classmethod
    @abstractmethod
    def _supports_regression(cls) -> bool:
        """
        Indicates if the Model is suitable for classification problems
        """
        pass

    @classmethod
    @abstractmethod
    def _supports_classification(cls) -> bool:
        """
        Indicates if the Model is suitable for classification problems
        """
        pass


class HDTreeClassifier(AbstractHDTree):
    def _check_preconditions(self):

        # check some general stuff
        status, msg = super()._check_preconditions()

        data = self.get_train_data()
        labels = self.get_train_labels()

        # Check if all guys are strings
        # ... or None (we want to gracefully support that by definition!)
        is_str = [isinstance(l, str) or l is None for l in labels]
        if not np.all(is_str):
            #status = False
            self._output_message("Warning: Labels for classification "
                                 "should to be of type String. Please explicitly cast if needed "
                                 "(e.g.: labels.astype(np.str)) "
                                 "I will go on, but random errors may occur (Code: 8342792)", only_if_verbose=False)

        # labels should not be too much. Basically checking if we are
        # really talking about a classification problem
        if len(np.unique(labels)) > len(data) * 0.8:
            status = False
            msg = "There seem to be unresonable many labels for classification (over 80%) Are you sure? (Code: 34723984)"

        if not self._information_measure.supports_classification():
            status = False
            msg = f"The given information measure has to support classification, " + \
                  f"however {self._information_measure.__class__.__name__} does not (Code: 3472389472938)"

        # check attributes (only if given)

        return status, msg

    def _supports_classification(cls):
        return True

    def _supports_regression(cls):
        return False

    def __str__(self):
        if not self.is_fit():
            return "Not fit tree. "

        return self.print_node(node=self._node_head, level=0, child_no=0)

    def print_node(self, node: Node, level: int, child_no: int) -> str:
        str = ''
        str += ("-" * level) + f"Level {level}, Child: {child_no}: {node}\n"
        childs = node.get_children()

        if childs:
            i = 1
            for child in childs:
                #str += f"{level}.{i}:\n"
                str += f'{self.print_node(node=child, level=level+1, child_no=i)}'
                i += 1

        return str

    def _get_prediction_for_node(self, node: Node) -> str:
        """
        Will return the value that a sample would be assigned if designated to that specific node
        :param node:
        :return:
        """
        labels = node.get_labels()
        return Counter(labels).most_common()[0][0]

    def _predict_sample(self, sample: np.ndarray) -> str:
        target_nodes = self._follow_for_sample_to_leafs(start_node=self._node_head, sample=sample)
        node_vals = []

        # retrieve value for ever node
        for target_node in target_nodes:
            node_vals.append((len(target_node.get_labels()), self._get_prediction_for_node(node=target_node)))

        sum_of_relevant_samples = sum([node_val[0] for node_val in node_vals])
        classes = [node_val[1] for node_val in node_vals]

        class_occurrences = Counter(classes).most_common()
        most_likely_class = class_occurrences[0][0]

        return most_likely_class

    def fit(self, X, y):
        super().fit(X=X, y=y)