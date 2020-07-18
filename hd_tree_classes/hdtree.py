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
from .split_rule import AbstractSplitRule
import logging
from collections import Counter
from graphviz import Digraph
import pandas as pd
from sklearn.metrics import accuracy_score


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
        self._node_head: Node = None
        self._min_samples_at_leaf = min_samples_at_leaf
        self._max_levels = max_levels
        self._allowed_splits = allowed_splits
        self._information_measure = information_measure
        self._verbose = verbose
        self._attribute_types: Optional[typing.List[str]] = None
        self._attribute_names = [*attribute_names] if attribute_names is not None else None
        self._is_fit = False

    def __copy__(self):
        """
        Will rebuilt the tree
        this operation is slow, since it will actually refit the model
        :return:
        """
        params = self.get_params()
        cpy_tree = self.__class__(**params)

        if not self.is_fit():
            return cpy_tree

        cpy_tree.prepare_fit(self.get_train_data(), self.get_train_labels())

        _head_node_cpy = self._node_head.__copy__()

        # follow all nodes
        nodes_to_expand = [_head_node_cpy]

        while len(nodes_to_expand) > 0:
            curr_node = nodes_to_expand.pop()

            # is it the head node?
            if curr_node.get_parent() is None:
                cpy_tree._node_head = curr_node

            # assign tree to copy
            curr_node.set_tree(cpy_tree)

            # get childs of the current node (if any)
            childs_of_copy = curr_node.get_children() or []

            # copy all children of the node + set current nodes childs to copies of childs
            child_copies = []
            for child_of_cpy in childs_of_copy:
                copy = child_of_cpy.__copy__()
                # children_cpy = [child.__copy__() for child in childs_of_copy]
                # copy.set_children(children_cpy)
                copy.set_parent(curr_node)
                if copy.get_split_rule() is not None:
                    copy.get_split_rule().set_node(curr_node)
                child_copies.append(copy)

            if len(child_copies) > 0:
                curr_node.set_children(child_copies)

            # append list to iterate through children
            nodes_to_expand += child_copies

        # prepent being fit already (we are, though parent tree may be fit)
        cpy_tree._is_fit = self.is_fit()

        #cpy.fit(self.get_train_data(), self.get_train_labels())
        return cpy_tree

    def map_attribute_indices_to_names(self, indices: typing.List[int]) -> typing.List[str]:
        """
        Will transform a list of attribute indices to its corresponding names
        :param indices:
        :return:
        """
        assert max(indices) < len(self.get_attribute_names()) and min(indices) >= 0, "Attribute indices out of bounds " \
                                                                                     "(Code: 3824728934)"
        return [*map(lambda idx: self.get_attribute_names()[idx], indices)]

    def set_max_level(self, max_levels: int):
        """
        Will set the maximum allowed levels of the tree,
        but will not change the current structure
        :param max_levels:
        :return:
        """
        assert 0 <= max_levels, "Maximum level cannot be negative (Code: 4583548934)"
        self._max_levels = max_levels

    def get_max_level(self) -> int:
        return self._max_levels

    def remove_node(self, node: Node):
        assert self.is_fit(), "The tree is not fit, so you cannot remove nodes (39482304823)"
        assert node is not self._node_head, "You cannot remove the head node (83293472)"

    def is_node_inside(self, node: Node) -> bool:
        """
        Checks if we find the
        :param node:
        :return:
        """
        nodes = self.get_all_nodes_below_node()
        return node in nodes

    def set_min_leaf_samples(self, min_samples: int):
        """
        Will set the minimum accepted  sample size at leafs but will
        not change the structure of the current tree
        :param min_samples:
        :return:
        """
        assert 1 <= min_samples, "Trees have to be able to at least grow to level 2 (Code: 984w7598354)"
        self._min_samples_at_leaf = min_samples

    def get_min_leaf_samples(self) -> int:
        return self._min_samples_at_leaf

    def get_params(self, deep=True):
        return {
            'allowed_splits': self._allowed_splits,
            'information_measure': self._information_measure,
            'attribute_names': self._attribute_names,
            'max_levels': self._max_levels,
            'min_samples_at_leaf': self._min_samples_at_leaf,
            'verbose': self._verbose,
        }

    def set_params(self, **kwargs):
        self._allowed_splits = kwargs['allowed_splits']
        self._information_measure = kwargs['information_measure']
        self._attribute_names = kwargs['attribute_names']
        self._max_levels = kwargs['max_levels']
        self._min_samples_at_leaf = kwargs['min_samples_at_leaf']
        self._verbose = kwargs['verbose']

    def get_allowed_splits(self) -> typing.List[typing.Type[AbstractSplitRule]]:
        """
        Gets a list of the currently allowed splits for that tree
        :return:
        """
        return self._allowed_splits

    def remove_allowed_split(self, split: typing.Type[AbstractSplitRule]):
        """
        Removes the given split type from allowed split
        if possible (otherwise does nothing)
        :param split:
        :return:
        """
        matches = [*map(lambda a_split: split is a_split, self.get_allowed_splits())]
        if any(matches):
            del self._allowed_splits[matches.index(True)]

    def add_allowed_split(self, split: typing.Type[AbstractSplitRule]):
        self._allowed_splits.append(split)

    def is_fit(self) -> bool:
        return self._is_fit

    def get_attribute_names(self) -> Optional[typing.List[str]]:
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

    def _check_predict_preconditions(self, X: np.ndarray):
        """
        Will check preconditions. If not met will raise
        :param X:
        :raises
        :return:
        """
        assert self.is_fit(), "Tree is fit on data, hence cannot predict (Code: 2842094823)"
        assert len(X.shape) == 2, "Data has to be in format n_samples x n_features (Code: 234234234)"
        assert X.shape[1] == len(self.get_attribute_names()), "Amount of labels has to match amount of " \
                                                              "features (Code: 23842039482)"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts samples in format [n_samples, n_features]
        :param X:
        :return:
        """

        self._check_predict_preconditions(X=X)
        ret_classes = []
        for sample in X:
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
        ret = f"Query: \n {sample_dict}\n\n"

        ret += f"Predicted sample as \"{self._predict_sample(sample=sample)}\" because of: \n"

        for i, child in enumerate(childs):
            path = [child]
            ret += f"Explanation {i + 1}:\n"

            while child.get_parent() is not None:
                path.append(child.get_parent())
                child = child.get_parent()

            path.reverse()

            for step, node in enumerate(path):
                ret += f"Step {step + 1}: {node.explain_split(sample=sample)}"
                if node.is_leaf():
                    ret += f' Vote for {self.get_possible_decisions_for_node(node=node)}'
                ret += '\n'
            ret += '---------------------------------\n'
        return ret

    def get_node_for_tree_walk(self, edge_indices: typing.List[int]) -> Node:
        """
        Will return the node that is the target after walking
        the edges specified

        [0,0,1,2] will follow left most node, left most node, second node, third node respectievely
        :param edge_indices:
        :return:
        """
        curr_node = self._node_head

        for i, edge_index in enumerate(edge_indices):
            childs = curr_node.get_children()
            assert childs is not None and edge_index >= 0 and edge_index < len(childs), \
                f"Node {curr_node} has no child {edge_index} or is a leaf" \
                f"(Walk step {i + 1}) (Code: 8473894753894)"
            curr_node = childs[edge_index]

        return curr_node

    def extract_node_chain_for_sample(self, sample: np.ndarray) -> typing.List[Node]:
        """
        Will extract all nodes from the head to the leaf the given sample
        falls into.
        It does not support tests on missing values at the moment
        :param sample:
        :return:
        """
        target_nodes = self._follow_for_sample_to_leafs(sample=sample, start_node=self._node_head)
        if len(target_nodes) > 1:
            raise Exception("Sorry, that samples belongs to more than one leaf node, I have no way to handle that at"
                            "the moment (Code: 384723894)")

        curr_node = target_nodes[0]
        nodes_chain = [curr_node]

        while curr_node.get_parent():
            curr_node = curr_node.get_parent()
            nodes_chain.append(curr_node)

        nodes_chain.reverse()

        return nodes_chain

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
                                 "Predictions will be the datas' priors", only_if_verbose=True)
        for child in childs:
            if child.is_leaf():
                leafs.append(child)
            else:
                childs += self._follow_for_sample_to_leafs(start_node=child,
                                                           sample=sample)

        return leafs

    def prepare_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Will prepare the tree to the data but not actually fit it.
        May be used to check if fit would work out
        :param X:
        :param y:
        :return:
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Data has to be in numpy array format!"
                            " Please use data.values if its a pandas DataFrame or np.ndarray(data)"
                            " if it is some Python iterable like a list (Code: 38420934)")
        self._labels = y
        self._train_data = X
        self._cached_predictions: typing.Dict = {}
        self._cached_uniques: typing.Dict[int, typing.Set] = {}
        self.classes_ = [*np.unique(y)]

        status, msg = self._check_preconditions()
        if status is False:
            raise Exception(f"Precondition check failed due to: \n\"{msg}\"\n(Code: 347239847239)")

    @abstractmethod
    def fit(self, X: np.ndarray,
            y: np.ndarray):
        """
        Will train the Tree
        :raises Exception if preconditions are not met
        """
        self.prepare_fit(X, y)

        # create head node over all indices
        self._node_head = self._create_node_for_data_indices(data_indices=[*range(0, len(self.get_train_data()))])

        current_nodes_to_split = [self._node_head]
        level = 0

        while True:
            self._output_message(f"Splitting level {level + 1}")
            collected_children = []

            for node_to_split in current_nodes_to_split:
                if self._max_levels is not None and self._max_levels == level:
                    break
                # n_node_samples = len(node_to_split.get_data_indices())

                # check if we want to split that node
                # we do not if we only have one sample or <= requested minimum amount of samples
                # if (self._min_samples_at_leaf is not None and self._min_samples_at_leaf >= n_node_samples) or \
                #  n_node_samples <= 1:
                #    continue

                # do not split if already clean
                if node_to_split.get_score() == 1.:
                    continue

                # split the node
                min_samples = 1 if not self._min_samples_at_leaf else self._min_samples_at_leaf
                self._split_node(node_to_split=node_to_split,
                                 level=level,
                                 min_samples_leaf=min_samples)

                # collect children of node the nodes if any
                children_of_node = node_to_split.get_children()
                if children_of_node:
                    collected_children += children_of_node

            current_nodes_to_split = collected_children
            level += 1

            # nothing left to split we are done
            if len(current_nodes_to_split) == 0:
                break

        self._is_fit = True

    def _split_node(self, node_to_split: Node, level: int, min_samples_leaf: int = 1):
        """
        Does a node split
        :param node_to_split:
        :param min_samples_leaf: ignore splits that produces leafs with less than that
        :param level: Current level we are within the corresponding tree (Counting from 0 for head)
        :return:
        """
        splits = []
        for split_type in self._allowed_splits:

            # check if we want to apply the rule to the level at all
            if level < split_type.get_min_level() or level > split_type.get_max_level():
                continue

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

        # get the best split
        # be aware that splits may be prioritized using their specificity value
        if len(splits) > 0:
            best_val = None
            best_split = None
            best_specificity = None
            for score_split in sorted(splits, key=lambda data: data[0], reverse=True):
                score, split = score_split
                # check if a more specific rule may have same score
                if best_val is None or score == best_val:
                    if best_specificity is None or split.get_specificity() > best_specificity:
                        best_val = score
                        best_split = split
                        best_specificity = split.get_specificity()
                else:  # since data is sorted if we reduce in performance we cannot do better -> leave to not eat more cycles
                    break

            node_to_split.set_split_rule(best_split)

    def get_all_nodes_below_node(self, node: Optional[Node]=None):
        """
        Will return all nodes under a given node
        if node is None will use head instead
        :param node:
        :return:
        """
        curr_node = node or self._node_head
        nodes = []
        childs = curr_node.get_children() or []

        for child in childs:
            nodes.append(child)
            nodes += self.get_all_nodes_below_node(node=child)

        return nodes

    def get_clean_nodes(self, min_score: float = 1., early_break: bool = True, node: Node = None):
        """
        Will return all nodes in the tree that have at least the given score

        :param min_score:
        :param node:
        :param early_break: Do not progress into childs if node meets requirement
        :return:
        """
        if node is None:
            curr_node = self._node_head
        else:
            curr_node = node

        clean_nodes = []

        childs = curr_node.get_children() or []

        for child in childs:
            progress = True
            if child.get_score() >= min_score:
                clean_nodes.append(child)
                progress = early_break

            if progress:
                clean_nodes += self.get_clean_nodes(min_score=min_score, node=child, early_break=early_break)

        return clean_nodes

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

        #numeric_kinds = {'b',  # boolean
        #                 'u',  # unsigned integer
        #                 'i',  # signed integer
        #                 'f',  # floats
        #                 'c'}

        for i in range(data.shape[1]):
            vals = data[:, i]

            # try to parse to float
            try:
                vals.astype(np.float)
                attribute = 'numerical'
            except ValueError:
                none_type = type(None)
                if np.all([isinstance(val, (str, none_type)) or np.isnan(val) for val in vals]):
                    attribute = 'categorical'

            #none_type = type(None)
            # numerical?
            #if vals.dtype.kind in numeric_kinds:
            #    attribute = 'numerical'
            # categorical?
            #elif np.all([isinstance(val, (str, none_type)) or np.isnan(val) for val in vals]):
            #    attribute = 'categorical'

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
            return False, f"Input data has to be in format (n_samples, n_features) but was {data.shape} (Code: 83472839472398)"

        if not len(labels.shape) == 1:
            return False, "Labels have to be in format (n_samples) (Code: 23874092374)"

        if not data.shape[0] > 0 or not data.shape[1] > 0:
            return False, "There has to be at least one sample and at least one Attribute (Code: 4723984723894)"

        # create attribute names if not given
        if self.get_attribute_names() is None:
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

    def generate_dot_graph(self, label_lookup: Optional[typing.Dict[any, str]] = None,
                           show_trace_of_sample: Optional[np.ndarray] = None) -> Digraph:
        """
        Returns the graphical representation of the tree.
        Can directly drawn into jupyter notebooks
        or saved to disk in a variety of file formats.
        Only works fully for classification at the moment

        :param label_lookup: replaces labels with given values for vizualization (like 0 -> No, 1 -> Yes)
        :param show_trace_of_sample: if sample is given the nodes this sample flows through will be marked visually
        :return:
        """
        assert self.is_fit(), "The decision tree is not fit, hence you cannot draw it (Code: 23489723489)"

        if show_trace_of_sample is not None:
            node_trace = self.extract_node_chain_for_sample(sample=show_trace_of_sample)
        else:
            node_trace = []

        # generate new dot environment
        dot = Digraph(comment='HDTree Export',
                      encoding="utf-8")

        def plot_one_node(node: Node, node_name: str):
            """
            Will plot one node with its inner parts
            :param node:
            :param node_name:
            :return:
            """
            description_text = f"\lSamples:      {node.get_sample_count()}" \
                               f"\lScore:        {round(node.get_score(), 2)}"

            rule = node.get_split_rule()
            if rule is not None:
                description_text += f"\lTest: {str.strip(str(rule))}"

            if self._supports_classification():
                labels = node.get_labels()
                if label_lookup:
                    labels = [*map(lambda lbl: label_lookup[lbl], labels)]
                description_text += '\n'
                labels_cnt = Counter(labels)
                most_common = labels_cnt.most_common()[0][0]
                for item in sorted(labels_cnt.items()):
                    name, amount = item
                    description_text += f"\l{name}: {amount}"
                    if most_common == name:
                        description_text += " ✓"

            # color-code clean nodes greenish
            cleaness = node.get_score()
            hex_number = hex(255 - int(cleaness * 255))[-2:]
            if hex_number[0] == "x":
                hex_number = '0' + hex_number[1]

            samples_total = len(self.get_train_labels())
            samples_in_node = len(node.get_data_indices())

            # proportion of all samples within tree that follow the given path
            proportion = samples_in_node / samples_total

            dot.node(node_name,
                     description_text + '\l ',
                     shape='box',
                     style="filled" if node not in node_trace else 'filled,diagonals',
                     fillcolor=f"#{hex_number}ff{hex_number}",
                     margin="0.3",
                     fontname="monospace",
                     penwidth=str(10 * proportion),
                     pencolor='#000000' if node not in node_trace else '#e69c43'
                     )

        # draw head node
        curr_nodes = [(self._node_head, 'Head (Level 0)')]
        plot_one_node(*curr_nodes[0])

        # draw all childs until no left
        node_number = 1
        level = 1
        while len(curr_nodes) > 0:
            new_nodes = []
            for node_name in curr_nodes:
                parent_node, parent_node_name = node_name
                childs = parent_node.get_children()

                edge_labels = parent_node.get_edge_labels()
                if childs is not None:
                    for child_num, child in enumerate(childs):
                        child_node_name = f'Node #{node_number} (Level {level})'
                        plot_one_node(node=child,
                                      node_name=child_node_name)

                        # scale arrows according to sample amount flowing through them
                        samples_parent = len(parent_node.get_data_indices())
                        samples_in_child = len(child.get_data_indices())
                        proportion = samples_in_child / samples_parent
                        dot.edge(parent_node_name, child_node_name,
                                 label=edge_labels[child_num],
                                 penwidth=str(7 * proportion))
                        new_nodes.append((child, child_node_name))
                        node_number += 1

            curr_nodes = new_nodes
            level += 1

        return dot

    @abstractmethod
    def get_prediction_for_node(self, node: Node,
                                force_recalculation: bool = False,
                                probabilistic: bool = False) -> typing.Union[str, typing.List[float]]:

        pass

    def get_possible_decisions_for_node(self, node: Node) -> typing.Set[str]:
        """
        Follows the path down to all leafs from the current node
        and returns all unique decisions that are found
        :return:
        """
        decisions = [self.get_prediction_for_node(node)]
        nodes_to_evaluate = [node]

        while len(nodes_to_evaluate) > 0:
            node = nodes_to_evaluate.pop()
            childs = node.get_children()
            if childs is not None:
                nodes_to_evaluate += childs
            else:
                decisions += [self.get_prediction_for_node(node)]

        return set(decisions)

    def simplify(self):
        """
        Will remove cut branches whose underlying decisions will not change no matter what leaf the sample might end up
        :return:
        """
        raise NotImplementedError()


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
            # status = False
            #self._output_message("Warning: Labels for classification "
            #                     "should to be of type String. Please explicitly cast if needed "
            #                     "(e.g.: labels.astype(np.str)) "
            #                     "I will go on, but random errors may occur (Code: 8342792)", only_if_verbose=False)
            pass

        # labels should not be too much. Basically checking if we are
        # really talking about a classification problem
        # if len(np.unique(labels)) > len(data) * 0.8:
        #    status = False
        #    msg = "There seem to be unresonable many labels for classification (over 80%) Are you sure? (Code: 34723984)"

        if not self._information_measure.supports_classification():
            status = False
            msg = f"The given information measure has to support classification, " + \
                  f"however {self._information_measure.__class__.__name__} does not (Code: 3472389472938)"

        # check attributes (only if given)

        return status, msg

    def __init__(self, *args, **kwargs):
        self.classes_: Optional[typing.List[str]] = None
        super().__init__(*args, **kwargs)

    @classmethod
    def _supports_classification(cls):
        return True

    @classmethod
    def _supports_regression(cls):
        return False

    def __str__(self):
        if not self.is_fit():
            return "Not fit tree. "

        return self.print_node(node=self._node_head, level=0, child_no=0)

    def print_node(self, node: Node, level: int, child_no: int, is_head=True) -> str:
        str = ''
        if not is_head:
            str += ("-" * level) + f"Level {level}, Child #{child_no}: {node}"
        else:
            str += f"Level {level}, ROOT: {node}"

        childs = node.get_children()
        if childs is None or len(childs) == 0:
            str += ' (LEAF)'

        str += '\n'

        if childs:
            i = 1
            for child in childs:
                # str += f"{level}.{i}:\n"
                str += f'{self.print_node(node=child, level=level + 1, child_no=i, is_head=False)}'
                i += 1

        return str

    def get_prediction_for_node(self, node: Node,
                                force_recalculation: bool = False,
                                probabilistic: bool = False) -> typing.Union[str, typing.List[float]]:
        """
        Will return the value that a sample would be assigned if designated to that specific node

        :param node:
        :param probabilistic: if true will return the |most_common| / |samples|
        :return:
        """

        if not probabilistic:
            is_cached = node in self._cached_predictions

            if not is_cached or force_recalculation:
                labels = node.get_labels()
                self._cached_predictions[node] = Counter(labels).most_common()[0][0]

            return self._cached_predictions[node]
        else:
            # list of popabilities is ordered by self.classes_
            labels = node.get_labels()
            rets: typing.List[float] = []
            vals = Counter(labels)

            for cls in self.classes_:
                rets.append(vals[cls] / len(labels))

            return rets

    def get_unique_values_for_attribute(self, attr_index: int) -> Optional[typing.Set]:
        """
        Returns unique values for an attribute (excluding None)
        if values are floating the function will return None

        if the uniques are proportionally more than half of the samples than we will not consider it valid
        for the Fixed Value Split, since it doesn't seem too categorical

        Same happens if we just have more than 50 Values

        @TODO make those two parameters parameters of the split and not static of the split
        :param attr_index: 
        :return: 
        """
        if attr_index not in self._cached_uniques:
            data = self.get_train_data()[:, attr_index]
            uniques = set(data)
            uniques.discard(None)
            count_non_none_elements = np.count_nonzero(data[~pd.isna(data)])
            #if attr_index == 2:
            #    print(count_non_none_elements, len(uniques), len(uniques) < 0.3 * count_non_none_elements or len(uniques) < 50)
            if len(uniques) < 0.3 * count_non_none_elements or len(uniques) < 50:
                self._cached_uniques[attr_index] = uniques
            else:
                self._cached_uniques[attr_index] = None

        return self._cached_uniques[attr_index]

    def _predict_sample(self, sample: np.ndarray,
                        probabilistic: bool = False) -> typing.Union[str, typing.List[float]]:

        target_nodes = self._follow_for_sample_to_leafs(start_node=self._node_head,
                                                        sample=sample)
        # if len(target_nodes) > 1:
        #    raise Exception("Predicting on missing values is not supported atm (Code: 23847238)")

        if not probabilistic:

            node_vals = []

            # retrieve values for every node
            for target_node in target_nodes:
                node_vals.append((target_node.get_sample_count(),
                                  self.get_prediction_for_node(node=target_node)))

            # sum_of_relevant_samples = sum([node_val[0] for node_val in node_vals])
            classes = [node_val[1] for node_val in node_vals]

            class_occurrences = Counter(classes).most_common()
            most_likely_class = class_occurrences[0][0]

            return most_likely_class
        else:
            return self.get_prediction_for_node(node=target_nodes[0], probabilistic=True)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def predict(self,
                X: np.ndarray,
                probabilistic: bool = False):

        self._check_predict_preconditions(X=X)

        if probabilistic:
            res = np.ndarray(shape=(len(X), len(self.classes_)), dtype=np.float)
        else:
            res = np.ndarray(shape=(len(X),), dtype=np.object)

        for i in range(len(X)):
            pred = self._predict_sample(
                sample=X[i],
                probabilistic=probabilistic)
            if probabilistic:
                res[i, :] = pred
            else:
                res[i] = pred

        if probabilistic:
            return res
        else:
            # having it as str from initialization doesnt work
            return res.astype(np.str)

    def predict_proba(self, X: np.ndarray):
        """
        Will return probabilistic results as list for each class ordered by self.classes_
        :param X:
        :return:
        """
        return self.predict(X=X, probabilistic=True)

    def fit(self, X, y):
        super().fit(X=X, y=y)

    def get_feature_count(self) -> int:
        """
        Returns the amount of features that this tree has access too (shape[1])
        :return:
        """
        return len(self.get_attribute_names())

    def get_all_nodes(self):
        """
        Will return a list of all nodes within tree

        :return:
        """
        assert self.is_fit(), "The tree is not fitted yet, hence has no Nodes inside (Code: 3298479823)"

        # list of nodes still in need to be expanded
        nodes_to_expand: typing.List[Node] = [self._node_head]

        # list of already expaned nodes
        expanded_nodes: typing.List[Node] = []

        while len(nodes_to_expand) > 0:
            # expand current nodes (get children)
            child_nodes: typing.List[Node] = []
            for node in nodes_to_expand:
                childs = node.get_children()
                if childs is not None:
                    child_nodes += childs
                    expanded_nodes.append(node)

            nodes_to_expand = child_nodes

        return expanded_nodes

    def compute_feature_importances(self) -> np.ndarray:
        """
        Tries to estimate the current classifiers feature importance
        using the increase of pureness given the current rules and used attributes
        sum of importances will be equal to 1.

        :return:
        """
        if not self.is_fit():
            raise Exception("Feature importance is evaluated using a fitted tree, please fit it first" \
                            " (Code: 3284723984)")

        importances = np.zeros(shape=(self.get_feature_count()), dtype=np.float)
        nodes = self.get_all_nodes()

        if len(nodes) > 0 and nodes[0].get_split_rule().is_initialized():
            max_samples = np.max([node.get_sample_count() for node in nodes])
            for node in nodes:
                if not node.is_leaf() or node.get_split_rule().is_initialized():
                    sample_cnt = node.get_sample_count()
                    score = node.get_score()
                    used_attributes = node.get_split_rule().get_split_attribute_indices()
                    child_score = self.get_information_measure().calculate_for_children(node)
                    # calculate how much that attribute(s) contributed to the increase in score
                    # (parent to childs)
                    relative_increase = (sample_cnt / max_samples) * (child_score - score)
                    importances[[*used_attributes]] += relative_increase

            importances /= np.sum(importances)

        else:
            importances[:] = 1. / len(importances)

        return importances

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Just a property that makes the tree sklearn compatible
        @see compute_feature_importances
        :return:
        """
        return self.compute_feature_importances()

    def simplify(self, return_copy: bool = False) -> 'HDTreeClassifier':
        """
        Will prune the tree until there are no branches left which would result in the same decisions
        independently from the sample

        :return:
        """

        me = self if not return_copy else self.__copy__()

        changed_happened = True
        while changed_happened:
            changed_happened=False
            leafs = [node for node in me.get_all_nodes_below_node(node=None) if node.is_leaf()]
            for leaf in leafs:
                parent = leaf.get_parent()

                # check if we try to cut head node
                if parent is not None:
                    decisions = me.get_possible_decisions_for_node(node=parent)
                    if len(decisions) == 1:
                        # we can could that guy
                        parent.make_leaf()
                        changed_happened = True

        return me

    @staticmethod
    def follow_node_to_root(node: Node) -> typing.List[Node]:
        """
        Will return the chain from root to current leaf
        in order root -> node (without node itself)

        :param node:
        :return:
        """
        ret = []
        parent = node.get_parent()
        while parent:
            ret.append(parent)
            parent = parent.get_parent()

        ret.reverse()

        return ret

    def get_head(self) -> Node:
        """
        Gets the first node inside
        :return:
        """
        return self._node_head

