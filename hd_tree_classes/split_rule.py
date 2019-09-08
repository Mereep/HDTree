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
import numpy as np
import pandas as pd
import logging


class AbstractSplitRule(ABC):
    """
    Represents one specific way
    a node may be split into child nodes
    """
   # from .node import Node
   # from .hdtree import AbstractHDTree

    def __str__(self):
        return self.user_readable_description()

    def __init__(self, node: 'Node'):
        self._node = node
        self._is_evaluated = False
        self._score: Optional[float] = None
        self._child_nodes: Optional[typing.List['Node']] = None
        self._score: Optional[float] = None
        self._state: Optional[typing.Dict[str, any]] = None

    def get_tree(self) -> 'AbstractHDTree':
        """
        Returns the tree that splitter belongs to
        """
        return self.get_node().get_tree()

    def get_score(self) -> Optional[float]:
        return self._score

    def get_information_measure(self) -> 'AbstractInformationMeasure':
        return self.get_tree().get_information_measure()

    def get_node(self) -> 'Node':
        return self._node

    def set_child_nodes(self, child_nodes: Optional[typing.List['Node']]):
        self._child_nodes = child_nodes

    @abstractmethod
    def user_readable_description(self) -> str:
        """
        Should explain the nodes split in textual form
        :return:
        """
        pass

    @abstractmethod
    def explain_split(self, sample: np.ndarray) -> str:
        """
        Returns human readable description for split decision
        :param sample:
        :return:
        """
        pass

    @abstractmethod
    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        """
        Will return the nodes that the sample hast to follow
        :param sample:
        :return:
        """
        assert self._is_evaluated, "This Rule was never fit on data (Code: 328420349823)"
        assert len(sample.shape) == 1, "Sample has to have exactly one dimension (Code: 2342344230)"
        assert len(sample) == len(self.get_tree().get_attribute_names()), "Amount of features does not match " \
                                                                          "trained features (Code: 34234234)"

        assert self.get_child_nodes() is not None and len(self.get_child_nodes()) > 1, "There are no child nodes " \
                                                                                       "attached (Code: 34234234)"

    @abstractmethod
    def _get_best_split(self) -> typing.Optional[typing.Tuple[float, typing.List['Node']]]:
        """
        Will return the best possible split over given Node using this split rule
        Has to return at least two Nodes

        may return None if split is not available for that Node
        """
        pass

    def get_child_nodes(self) -> Optional[typing.List['Node']]:
        """
        Gets child nodes, if any
        """
        return self._child_nodes

    def __call__(self) -> Optional[float]:
        """
        Will evaluate the criteroin on all splits
        :raises Exception if called twice or rule did not return at least two child nodes
        """

        if self._is_evaluated:
            raise Exception("Split can only be evaluated once per node (Code: 32472389472)")

        # set us self as split rule for the moment (we set this back)
        old_split = self.get_node().get_split_rule()
        self.get_node().set_split_rule(self)
        ret = self._get_best_split()
        self.get_node().set_split_rule(rule=old_split)

        score = None
        if ret:
            score, nodes = ret
            if len(nodes) < 2:
                raise Exception("A Node has to split into at least two children (Code: 32487239874)")

            self._score = score
            self.set_child_nodes(child_nodes=nodes)

        self._is_evaluated = True

        return score

    @abstractmethod
    def _get_attribute_candidates(self) -> typing.List[int]:
        """
        Returns a list attribute / feature indices that Split Rule is apllicable at
        """
        pass

    @classmethod
    def _filter_col_none(cls, column: np.ndarray):
        """
        returns als members of column that are not none
        """
        assert len(column.shape) == 1, "Column is to be one-dimensional (Code: 974298473)"

        return column[~cls._get_col_none_indexer(column=column)]

    @staticmethod
    def _get_col_none_indexer(column: np.ndarray) -> np.ndarray:
        """
        Returns a numpy boolean indexer having non indices being True otherwise false
        :param column:
        :return:
        """
        assert len(column.shape) == 1, "Column is to be one-dimensional (Code: 2345675432)"

        return pd.isnull(column)

    def set_state(self, state: typing.Dict[str, any]):
        """
        Will set the rules internal state
        that carries the information to split the node
        :param state:
        :return:
        """
        self._state = state

    def get_state(self) -> Optional[typing.Dict[str, any]]:
        return self._state


class AbstractNumericalSplit(AbstractSplitRule):
    """
    Represents a split over categorical attributes
    """

    def _get_attribute_candidates(self) -> typing.List[int]:
        """
        Returns a list of attribute / feature indices that Split Rule is apllicable at
        """
        candidates = []
        for i, t in enumerate(self.get_tree().get_attribute_types()):
            if t == 'numerical':
                candidates.append(i)

        return candidates



class AbstractCategoricalSplit(AbstractSplitRule):
    """
    Represents a split over numerical attributes
    """

    def _get_attribute_candidates(self) -> typing.List[int]:
        """
        Returns a list of attribute / feature indices that Split Rule is apllicable at
        """
        candidates = []
        attr_types = self.get_tree().get_attribute_types()
        for i, t in enumerate(self.get_tree().get_attribute_types()):
            if t == 'categorical':
                candidates.append(i)

        return candidates


class TwoAttributeSplitMixin(AbstractSplitRule):
    """
    This is just a helper that deals with extracting the necessary data (including none indices)
    over exactly two attributes to easen up some repeated works
    """
    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        data_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) < 2:
            return None

        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        best_state = {}

        # iterate over all attributes that are available
        # will iterate over two attribute indices where each pair is only regarded exactly once
        for i in range(len(supported_cols)):
            for j in range(i, len(supported_cols)):
                attr_idx1 = supported_cols[i]
                attr_idx2 = supported_cols[j]
                col_data_1 = node_data[:, attr_idx1]
                col_data_2 = node_data[:, attr_idx2]
                null_indexer_1 = self._get_col_none_indexer(column=col_data_1)
                null_indexer_2 = self._get_col_none_indexer(column=col_data_2)
                non_null_indexer_1 = ~null_indexer_1
                non_null_indexer_2 = ~null_indexer_2
                non_null_indexer_1_2 = non_null_indexer_1 & non_null_indexer_2

                if np.any(non_null_indexer_1_2):
                    # get indexer for child node data assignments
                    rets = self._get_children_indexer_and_state(data_values_left=col_data_1[non_null_indexer_1_2],
                                                               data_values_right=col_data_2[non_null_indexer_1_2])

                    if not isinstance(rets, typing.List):
                        logging.getLogger(__package__).warning(f"Returning only one split for a split rule "
                                                               f"is deprecated"
                                                               f"but {self.__class__.mro()[0]} does."
                                                               f"if you just want to return a single split, "
                                                               f"just return [(config, [nodes...])]. Fixing "
                                                               f"that for you"
                                                               f"Code (23180198123)")
                        rets = [rets]
                    curr_child_indices = []

                    # a valid split occured?
                    for ret in rets:
                        if ret is not None:
                            null_indexer_1_2 = ~non_null_indexer_1_2
                            non_null_indices = data_indices[non_null_indexer_1_2]
                            has_null_entries = np.any(null_indexer_1_2)
                            state, assignments = ret

                            assert isinstance(state, typing.Dict),        "State mus be a dict (Code: 67543256)"
                            assert isinstance(assignments, typing.Tuple), "Returned data has to be " \
                                                                          "a tuple or None (Code: 5869378963547)"
                            assert len(assignments) >= 2,                 "A split has to generate at least " \
                                                                          "two child nodes(" \
                                                                          "Code: 45698435673)"

                            # collect data indices for each node
                            has_empty_child = False
                            for child_assignment in assignments:
                                assert isinstance(child_assignment, np.ndarray) and child_assignment.dtype is np.dtype(np.bool), \
                                    "Assignments have to be a numpy array of type bool (Code: 39820934)"

                                if not np.any(child_assignment):
                                    has_empty_child = True
                                    break

                                curr_child_indices.append(non_null_indices[child_assignment])

                                if has_null_entries:
                                    curr_child_indices[-1] = np.append(curr_child_indices[-1],
                                                                       data_indices[null_indexer_1_2])

                            # check if we have an empty child
                            # atm we do not want to accept that -> ignore
                            if not has_empty_child:
                                child_nodes = [node.create_child_node(assigned_data_indices=indices)
                                               for indices in curr_child_indices]

                                state['split_attribute_indices'] = (attr_idx1, attr_idx2)
                                self.set_child_nodes(child_nodes=child_nodes)
                                self.set_state(state)
                                score = self.get_information_measure()(parent_node=self.get_node())

                                if score > best_score:
                                    best_score = score
                                    best_children = self.get_child_nodes()
                                    best_state = state

                # check if we scored something at all
                if best_score > -float('inf'):
                    self.set_state(best_state)
                    return best_score, best_children

        # otherwise we just don't have something to show
        return None

    @abstractmethod
    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray) -> Optional[typing.Tuple[typing.Dict,
                                                                                                typing.Tuple[np.ndarray]]]:
        """
        Returns for each child no
        :param data_values_left:
        :param data_right:
        :return: state, tuple of child indices
        """
        pass

    def get_split_attribute_indices(self) -> typing.Tuple[int, int]:
        """
        Will return the used attributes' split index
        raises some Exception if split is not initialized
        :return:
        """
        assert self.get_state() is not None, "Split is not initialized, hence it has not split attribute " \
                                             "(Code: 34233756446)"
        state = self.get_state()
        if 'split_attribute_indices' not in state:
            raise Exception("There is no valid split index set in state, this is a programming error! (Code: 3422534)")

        return self.get_state()['split_attribute_indices']

    def get_split_attribute_names(self) -> typing.Tuple[str, str]:
        """
        Just a convenience method that returns the names of the split attribute
        :return:
        """
        indices = self.get_split_attribute_indices()
        return (self.get_tree().get_attribute_names()[indices[0]],
                self.get_tree().get_attribute_names()[indices[1]])

class OneAttributeSplitMixin(AbstractSplitRule):
    """
    This is just a helper that deals with extracting the necessary data (including none indices)
    to easen up some repeated works

    """

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        data_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        best_state = {}

        # iterate over all attributes that are available
        for attr_idx in supported_cols:
            col_data = node_data[:, attr_idx]
            null_indexer = self._get_col_none_indexer(column=col_data)
            non_null_indexer = ~null_indexer

            if np.any(non_null_indexer):
                non_null_values = col_data[non_null_indexer]

                # get indexer for child node data assignments
                rets = self._get_children_indexer_and_state(data_values=non_null_values)

                curr_child_indices = []

                # a valid split occured?
                if rets is not None:
                    # we simulate old behaviour if we get no list of stuffies back
                    # = when only one result is returned
                    if not isinstance(rets, typing.List):
                        logging.getLogger(__package__).warning(
                            f"Split \"{self.__class__.__mro__[0].__name__}\" rule did not return a list of "
                            "dict, split-pairs"
                            "we simulated that for now. If you only "
                            "want to return exactly"
                            "one split just put brackets around it, "
                            "e.g.: \"[]\" (Code: 3249823094823)")
                        rets = [rets]
                    for ret in rets:


                        non_null_indices = data_indices[non_null_indexer]
                        has_null_entries = np.any(null_indexer)
                        state, assignments = ret

                        assert isinstance(state, typing.Dict),        "State mus be a dict (Code: 32942390)"
                        assert isinstance(assignments, typing.Tuple), "Returned data has to be " \
                                                                      "a tuple or None (Code: 00756435345)"
                        assert len(assignments) >= 2,                 "A split has to generate at least two child nodes (" \
                                                                      "Code: 248234)"

                        # collect data indices for each node
                        has_empty_child = False
                        for child_assignment in assignments:
                            assert isinstance(child_assignment, np.ndarray) and child_assignment.dtype is np.dtype(np.bool), \
                                "Assignments have to be a numpy array of type bool (Code: 39820934)"

                            if not np.any(child_assignment):
                                has_empty_child = True
                                break

                            curr_child_indices.append(non_null_indices[child_assignment])

                            if has_null_entries:
                                curr_child_indices[-1] = np.append(curr_child_indices[-1], data_indices[null_indexer])

                        # check if we have an empty child
                        # atm we do not want to accept that -> ignore
                        if not has_empty_child:
                            child_nodes = [node.create_child_node(assigned_data_indices=indices)
                                           for indices in curr_child_indices]

                            state['split_attribute_indices'] = [attr_idx]
                            self.set_child_nodes(child_nodes=child_nodes)
                            self.set_state(state)
                            score = self.get_information_measure()(parent_node=self.get_node())

                            if score > best_score:
                                best_score = score
                                best_children = self.get_child_nodes()
                                best_state = state

        # check if we scored something at all
        if best_score > -float('inf'):
            self.set_state(best_state)
            return best_score, best_children

        # otherwise we just dont have something to show
        return None

    @abstractmethod
    def _get_children_indexer_and_state(self, data_values: np.ndarray) -> Optional[typing.List[typing.Tuple[typing.Dict,
                                                                                                typing.Tuple[np.ndarray]]]]:
        """
        Returns for each child no
        :param data_values:
        :param data_indices:
        :return: state, tuple of child indices
        """
        pass

    def get_split_attribute_index(self) -> int:
        """
        Will return the used attributes split index
        raises some Exception if split is not initialized
        :return:
        """
        assert self.get_state() is not None, "Split is not initialized, hence it has not split attribute " \
                                             "(Code: 34233756446)"
        state = self.get_state()
        if 'split_attribute_indices' not in state:
            raise Exception("There is no valid split index set in state, this is a programming error! (Code: 3422534)")

        return self.get_state()['split_attribute_indices'][0]

    def get_split_attribute_name(self) -> str:
        """
        Just a convinience method that returns the name of the split attribute
        :return:
        """
        idx = self.get_split_attribute_index()
        return self.get_tree().get_attribute_names()[idx]


class NumericalSplit(AbstractNumericalSplit, OneAttributeSplitMixin):

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_name = self.get_split_attribute_name()
            split_val = state['split_value']
            attr_index = self.get_split_attribute_index()

            attr = sample[attr_index]

            if attr is None:
                return f"Attribute \"{attr_name}\" is not available"
            else:
                if attr < split_val:
                    return f"\"{attr_name}\" < " \
                           f"{round(split_val, 2)}"
                else:
                    return f"\"{attr_name}\" ≥ " \
                           f"{round(split_val, 2)}"
        else:
            raise Exception("Numerical split not initialized, hence, cannot explain a decision (Code: 8903485768930)")

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()

        if state is not None:
            attr_name = self.get_split_attribute_name()
            split_val = state['split_value']
            return f"{attr_name} < " \
                   f"{round(split_val, 2)}"
        else:
            return "Numerical split not initialized"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        state = self.get_state()
        val = state['split_value']
        attr_idx = self.get_split_attribute_index()

        attr = sample[attr_idx]
        if attr is None:
            return self.get_child_nodes()
        else:
            if attr < val:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_children_indexer_and_state(self, data_values: np.ndarray):
        split_member = np.median(data_values)
        left = data_values < split_member
        right = ~left
        state = {'split_value': split_member}

        return [(state, (left, right))]


class CloseToMedianSplit(AbstractNumericalSplit, OneAttributeSplitMixin):
    """
    Will split on a numerical attributes median +- 1/2 * stdev
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()

        if state is not None:
            attr_index = self.get_split_attribute_index()
            attr_name = self.get_split_attribute_name()
            val = sample[attr_index]
            median = state['median']
            stddev = state['stdev']

            if val is None:
                return f"\"Value for {attr_name}\" is not available, hence assigned to all children"
            else:
                if abs(val - median) <= 0.5 * stddev:
                    return f"\"{attr_name}\" is close to groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ = {0.5 * round(stddev, 2)}), hence assigned to left child"
                else:
                    return f"\"{attr_name}\" is outside of groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ = {0.5 * round(stddev, 2)}), hence assigned to right child"
        else:
            raise Exception("Close To Median Split not initialized, hence, "
                            "cannot explain a decision (Code: 234678234902347)")

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()

        if state is not None:
            attr_name = self.get_split_attribute_name()
            median = state['median']
            stdev = state['stdev']

            return f"{attr_name} is close to groups' median of " \
                   f"{median} (± ½ × σ = {0.5 * round(stdev, 2)})?"
        else:
            return "Close To Median Split is not initialized"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        state = self.get_state()
        if state is not None:
            attr_idx = self.get_split_attribute_index()
            median = state['median']
            stdev = state['stdev']
            val = sample[attr_idx]

            if val is None:
                return self.get_child_nodes()
            else:
                if abs(median - val) <= 0.5 * stdev:
                    return [self.get_child_nodes()[0]]
                else:
                    return [self.get_child_nodes()[1]]

    def _get_children_indexer_and_state(self, data_values: np.ndarray):

        # get median val and standard deviation
        median_val = np.median(data_values)
        stdev = np.std(data_values)

        # everything close to median by means of stddev goes to left node
        inside_median = np.abs(data_values - median_val) <= 0.5 * stdev

        state = {'median': median_val, 'stdev': stdev}

        return [(state, (inside_median, ~inside_median))]


class SmallerThanSplit(AbstractNumericalSplit, TwoAttributeSplitMixin):
    """
    Splits on a1 < a2 rule
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attributes: typing.Tuple[int, int] = None

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()
        if state is not None:
            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            return f"{attr_name_1} < " \
                   f"{attr_name_2}"
        else:
            return "Smaller than split not initialized"

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute \"{attr_name_1}\" is not available, hence assigned to both children"
            elif attr2 is None:
                return f"Attribute \"{attr_name_2}\" is not available, hence assigned to both children"
            else:
                if attr1 < attr2:
                    return f"\"{attr_name_1}\" < \"{attr_name_2}\""
                else:
                    return f"\"{attr_name_1}\" ≥ \"{attr_name_2}\""

        else:
            raise Exception("Smaller Than Split not initialized, hence, "
                            "cannot explain a decision (Code: 234237423987423)")

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]

        if attr1 is None or attr2 is None:
            return self.get_child_nodes()
        else:
            if attr1 < attr2:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray):
        left_vals = data_values_left < data_values_right
        state = {}

        return [(state, (left_vals, ~left_vals))]


class LessThanHalfOfSplit(AbstractNumericalSplit, TwoAttributeSplitMixin):
    """
    Splits on a1 < 1/2 * a2 rule
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attributes: typing.Tuple[int, int] = None

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()
        if state is not None:
            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            return f"{attr_name_1} is less than half of " \
                   f"{attr_name_2}"
        else:
            return "Smaller than split not initialized"

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute \"{attr_name_1}\" is not available, hence assigned to both children"
            elif attr2 is None:
                return f"Attribute \"{attr_name_2}\" is not available, hence assigned to both children"
            else:
                if attr1 < 0.5 * attr2:
                    return f"\"{attr_name_1}\" < ½ × {attr_name_2}"
                else:
                    return f"\"{attr_name_1}\" ≥ ½ × {attr_name_2}"

        else:
            raise Exception("Less Than Half Of Split not initialized, hence, "
                            "cannot explain a decision (Code: 23423234234234)")

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]

        if attr1 is None or attr2 is None:
            return self.get_child_nodes()
        else:
            if attr1 < 0.5 * attr2:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray):
        left_vals = data_values_left < 0.5 * data_values_right
        state = {}

        return [(state, (left_vals, ~left_vals))]


class SingleCategorySplit(AbstractCategoricalSplit, OneAttributeSplitMixin):
    """
   Will split on a single categorical attribute
   e.g. an attribute having 10 unique values will split in up tp 10 childs
   (Depending if node in question has all values inside)
   """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()
        if state is not None:
            attr_name = self.get_split_attribute_name()
            lookup = state['label_to_node_idx_lookup']
            print(lookup.keys())
            return f"Split on categorical attribute \"{attr_name}\" with possible values: {', '.join([*lookup.keys()])}"
        else:
            return "Single Category Split is not evaluated"

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx = self.get_split_attribute_index()
            attr_name = self.get_split_attribute_name()
            attr = sample[attr_idx]
            lookup = state['label_to_node_idx_lookup']

            if attr is None:
                return f"\"{attr_name}\" is not available, hence assigned to all child nodes"
            elif attr not in lookup:  # we did not split on that specific val
                return f"\"{attr_name}\" with value {attr} was not available at that stage during training, " \
                       f"hence assigned to all childs"
            else:
                attr_node = lookup[attr]
                return f"\"{attr_name}\" is {attr}, hence assigned it to node number {attr_node+1}"

        else:
            raise Exception("Single Category Split Split not initialized, hence, "
                            "cannot explain a decision (Code: 365345673459683456)")

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        state = self.get_state()
        lookup = state['label_to_node_idx_lookup']
        attr_idx = self.get_split_attribute_index()
        val = sample[attr_idx]

        # return all nodes if we encounter an invalid value´(at training time) or a None
        if val is None or val not in lookup:
            return self.get_child_nodes()
        else:
            return [self.get_child_nodes()[lookup[val]]]

    def _get_children_indexer_and_state(self, data_values: np.ndarray):

        distinct_node_labels = np.unique(data_values)
        node_indexer = []
        label_to_node_idx_lookup = {}

        # only split if we have at least two distinct values
        if len(distinct_node_labels) >= 2:
            for i, lbl in enumerate(distinct_node_labels):
                node_indexer.append(data_values == lbl)
                label_to_node_idx_lookup[lbl] = i

            return [({'label_to_node_idx_lookup': label_to_node_idx_lookup},\
                   tuple(node_indexer))]

        return None




