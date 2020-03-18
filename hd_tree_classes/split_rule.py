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

from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
import logging
import json
import sys
import inspect
from functools import wraps
import typing
from typing import Optional, Union, Tuple, List, Dict

# ANNOTATIONS
def hide_in_ui(cls: 'AbstractSplitRule') -> 'AbstractSplitRule':
    cls._hide_in_ui = True
    return cls

def check_initialized(f: typing.Callable, *args, **kwargs):
    """
    Will check if the split is initialized before calling the function
    and raise if not
    :param f:
    :param args:
    :param kwargs:
    :return:
    """
    @wraps(f)
    def check(self: 'AbstractSplitRule', *args, **kwargs):
        if not self.is_initialized():
            raise Exception(f"Could not perform {f.__name__}. The split {self.get_name()} "
                            f"needs to be initialized (Code: 38472384723" )
        return f(self, *args, **kwargs)

    return check


class AbstractSplitRule(ABC):
    """
    Represents one specific way
    a node may be split into child nodes
    """

    _min_level: int = 0               # minimum level this rule may be applied from
    _max_level:int = sys.maxsize      # maximum level that rule may be applied to

    # restrict to the given attributes
    _whitelist_attribute_indices: Optional[typing.List[typing.Union[str, int]]] = None

    # remove application of the rule from these
    _blacklist_attribute_indices: Optional[typing.List[typing.Union[str, int]]] = None

    _state: Optional[Dict[str, any]] = None

    _hide_in_ui: bool = False

    @classmethod
    def new_from_other(cls, other: 'AbstractSplitRule', state: Dict) -> 'AbstractSplitRule':
        """
        Will create the rule like it would be in-place with the other rule
        :param other:
        :param state:
        :return:
        """
        node = other.get_node()
        new = cls(node=node)
        state['split_attribute_indices'] = other.get_state()['split_attribute_indices']
        new._state = state
        new._is_evaluated = True
        new._child_nodes = other._child_nodes

        return new

    @classmethod
    def show_in_ui(cls) -> bool:
        return not cls._hide_in_ui


    def __str__(self):
        return self.user_readable_description()

    @classmethod
    def get_help_text(cls):
        """
        General help text
        :return:
        """
        return "A splitting rule divides a sample into two or more subgroups"

    def get_help_text_instance(self):
        """
        Will return a help text specific to the instance (may or may nor differ from @see get_help_text)
        :return:
        """
        return self.get_help_text()

    @classmethod
    def get_min_level(cls) -> int:
        return cls._min_level

    @classmethod
    def get_max_level(cls) -> int:
        return cls._max_level

    @classmethod
    def set_min_level(cls, val: int):
        cls._min_level = val

    @classmethod
    def set_max_level(cls, val: int):
        cls._max_level = val

    @classmethod
    def clone_class_type(cls) -> typing.Type['AbstractSplitRule']:
        """
        Creates a new type from the current rule (class-type)
        :return:
        """
        cls_me = type(cls.get_name(), (cls, ), {})
        return cls_me

    @classmethod
    def clear_whitelist_entries(cls):
        """
        Removes all whitelist entries (if any), be aware that this affects all instances of that class
        :return:
        """
        cls._whitelist_attribute_indices = []

    @classmethod
    def clear_blacklist_entries(cls):
        """
        Removes all whitelist entries (if any), be aware that this affects all instances of that class
        :return:
        """
        cls._blacklist_attribute_indices = []

    @classmethod
    def add_whitelist_entry(cls, attr: int):
        """
        Adds a whitelist attribute. Be aware that this changes all derived instances for this rule

        :param attr:
        :return:
        """
        assert not cls.has_blacklist_attributes(), "This rule has alread blacklist attributes, hence cannot add " \
                                                   "whitelist attributes (Code: 832472389)"

        if not cls.has_whitelist_attributes():
            cls._whitelist_attribute_indices = []

        assert attr not in cls.get_whitelist_attribute_indices(), "The attribute IS already in " \
                                                                  "whitelist (Code: 87345928795)"

        cls._whitelist_attribute_indices.append(attr)

    @classmethod
    def add_blacklist_entry(cls, attr: int):
        """
        Adds a blacklist attribute. Be aware that this changes all derived instances for this rule

        :param attr:
        :return:
        """
        assert not cls.has_whitelist_attributes(), "This rule has alread whitelist attributes, hence cannot add " \
                                                   "blacklist attributes (Code: 5675467986)"

        if not cls.has_blacklist_attributes():
            cls._blacklist_attribute_indices = []

        assert attr not in cls.get_blacklist_attribute_indices(), "The attribute IS already in blacklist " \
                                                                  "(Code: 54638973458)"

        cls._blacklist_attribute_indices.append(attr)   \

    @classmethod
    def remove_blacklist_entry(cls, attr: int):
        """
        Removes a blacklist attribute. Be aware that this changes all derived instances for this rule

        :param attr:
        :return:
        """
        assert cls.has_blacklist_attributes(), "This rule has not blacklist attributs, hence cannot remove " \
                                               "blacklist attributes (Code: 5434543)"

        assert attr in cls.get_blacklist_attribute_indices(), "The attribute IS NOT in blacklist (Code: 28263324345673)"

        del cls._blacklist_attribute_indices[cls._blacklist_attribute_indices.index(attr)]

    @classmethod
    def remove_whitelist_entry(cls, attr: int):
        """
        Removes a whitelist attribute. Be aware that this changes all derived instances for this rule

        :param attr:
        :return:
        """
        assert cls.has_whitelist_attributes(), "This rule has not whitelist attributs, hence cannot remove " \
                                               "whitelist attributes (Code: 4564564564556843)"

        assert attr in cls.get_whitelist_attribute_indices(), "The attribute IS NOT in whitelist (Code: 43576354345)"

        del cls._whitelist_attribute_indices[cls._whitelist_attribute_indices.index(attr)]

    @classmethod
    def get_name(self) -> str:
        return self.__mro__[0].__name__

    @property
    def _is_evaluated(self) -> bool:
        return self.is_initialized()

    @_is_evaluated.setter
    def _is_evaluated(self, value=None):
        pass #  This is a noop just to make it backwards compatible

    def __init__(self, node: 'Node'):
        self._node = node
        self._is_evaluated = False
        self._score: Optional[float] = None
        self._child_nodes: Optional[typing.List['Node']] = None
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
    def get_edge_labels(self) -> typing.List[str]:
        """
        Returns short labels that can be used as a edge texts
        justifying the direction
        :return:
        """
        pass

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

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        """
        Will return the nodes that the sample has to follow
        :param sample:
        :return:
        """
        indices = self.get_child_node_indices_for_sample(sample=sample)
        return [node for i, node in enumerate(self.get_child_nodes()) if i in indices]

    @abstractmethod
    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List[int]:
        """
        Returns the indices of the child node that sample would be assigned to
        :param sample:
        :return:
        """
        assert self._is_evaluated, "This Rule was never fit on data (Code: 328420349823)"
        assert len(sample.shape) == 1, "Sample has to have exactly one dimension (Code: 2342344230)"
        assert len(sample) == len(self.get_tree().get_attribute_names()), "Amount of features does not match " \
                                                                          "trained features (Code: 34234234)"

        assert self.get_child_nodes() is not None and len(self.get_child_nodes()) > 1, "There are no child nodes " \
                                                                                       "attached (Code: 4564567)"

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
        Will evaluate the criterion on all splits
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

    @classmethod
    def get_min_level(cls) -> int:
        """
        Rule should only be applied beginning from that level
        :return:
        """
        return cls._min_level

    @classmethod
    def get_max_level(cls) -> int:
        """
        Rule should only be applied beginning from that level
        :return:
        """
        return cls._max_level


    @classmethod
    def get_whitelist_attribute_indices(cls) -> Optional[typing.List[int]]:
        """
        If given, the rule may ONLY be applied to these indices
        if its none we whitelist every attribute
        :return:
        """
        return cls._whitelist_attribute_indices

    @classmethod
    def get_blacklist_attribute_indices(cls) -> Optional[typing.List[int]]:
        """
        If given, the rule may NOT be applied to these indices
        if its none we blacklist nothing
        :return:
        """
        return cls._blacklist_attribute_indices

    @classmethod
    def build_with_restrictions(cls,
                                whitelist_attribute_indices: Optional[typing.List[typing.Union[str, int]]]=None,
                                blacklist_attribute_indices: Optional[typing.List[typing.Union[str, int]]]=None,
                                min_level: int = 0,
                                max_level: int = sys.maxsize):
        """
        Will generate the rule class as type with given restrictions
        :param whitelist_attribute_indices: If set, the rule will ONLY be applied to indices not involving the given list
        :param blacklist_attribute_indices: If set, the rule will NOT be applied to indices not involving the given list
        :param min_level: Rule will not be applied before this tree level (starting from 0 for head)
        :param max_level: Will not be applied AFTER that rule (including)
        :return:
        """
        assert min_level >= 0 and min_level <= max_level, "0 <= min_level <= max_level (Code: 8347982374)"
        assert not (blacklist_attribute_indices is not None and whitelist_attribute_indices is not None), \
            "Blacklist and whitelist cannot be used together. If you want both ways, " \
            "just add a second rule (Code: 213123231)"

        t = type(cls.__mro__[0].__class__.__name__, (cls,), {
            '_whitelist_attribute_indices': whitelist_attribute_indices,
            '_blacklist_attribute_indices': blacklist_attribute_indices,
            '_min_level': min_level,
            '_max_level': max_level
        })

        return t

    def _check_rule_applicable_to_attribute_index(self, index: int):
        """
        Check if whitelist or blacklist restrict access to that attribute
        :param index:
        :return:
        """
        whitelist = self.get_whitelist_attribute_indices()
        blacklist = self.get_blacklist_attribute_indices()

        has_whitelist = self.has_whitelist_attributes()
        has_blacklist = self.has_blacklist_attributes()
        if self.get_tree() is not None:
            if (has_whitelist and isinstance(whitelist[0], str)) or \
                    (has_blacklist and isinstance(blacklist[0], str)):

                index = self.get_tree().get_attribute_names()[index]

        if has_whitelist:
            return index in whitelist

        if has_blacklist:
            return index not in blacklist

        return True

    def get_state(self) -> Optional[typing.Dict[str, any]]:
        return self._state

    @classmethod
    def has_whitelist_attributes(self) -> bool:
        return self._whitelist_attribute_indices is not None and len(self._whitelist_attribute_indices) > 0

    @classmethod
    def has_blacklist_attributes(self) -> bool:
        return self._blacklist_attribute_indices is not None and len(self._blacklist_attribute_indices) > 0

    def is_initialized(self) -> bool:
        """
        Checks if the split tule is in initialized state
        :return:
        """
        return self.get_state() is not None

    @abstractmethod
    def get_split_attribute_indices(self) -> typing.Tuple[int]:
        """
        Returns the involved split attributes
        :return:
        """
        pass

    def __add__(self, other: 'AbstractSplitRule', use_attribute_names: bool=True,
                sample: Optional[np.ndarray] = None, try_reverse: bool=True) -> Optional[Union['AbstractSplitRule',
                                                                    Tuple['AbstractSplitRule',
                                                                          'AbstractSplitRule']]]:
        """
        Will try to Merge to split rules.
        If success it will return ONE split rule that is combining the both rules, otherwise will return
        the ORIGINAL splits

        Attention merging rules based on different indice numbers will return merged rules not matching the
        indices of both original rules at the same time!
        -
        :param other:
        :param use_attribute_names: if set to True it will not availuate by attribute indices but will fetch their names
                                    and merge them
        :return:
        """
        if self.is_initialized():
            if not use_attribute_names or self.get_node() is None:
                # merge if indices are the same, note if there is
                # no tree option attribute names is silently over-writen
                can_merge = self.get_split_attribute_indices() == other.get_split_attribute_indices()
            else: # merge if names are the same
                split_names_left = [self.get_tree().get_attribute_names()[idx] for idx in self.get_split_attribute_indices()]
                split_names_right = [other.get_tree().get_attribute_names()[idx] for idx in other.get_split_attribute_indices()]
                can_merge = split_names_left == split_names_right

            if can_merge:
                res = self._merge(other, sample=sample)
                if res is None and try_reverse:
                    # try other way around
                    res = other._merge(self, sample=sample)

                if res is not None:
                    return res

            return self, other
        else:
            raise Exception("You can only combine initialized rules (Code: 328749823)")

    def _merge(self, other, sample: Optional[np.ndarray] = None) -> Optional['AbstractSplitRule']:
        """
        If two rules can be reduced, method should return a rule that combines these two
        don't use this function directly, use +-Operator instead

        If sample is not given, a rule that satisfied both conditions is to be created.
        Otherwise a minimal rule that satisfied both conditions related to the rule has to be returned

        :param other:
        :param sample: if given rule should reflect the concrete decision for this sample,
        :return:
        """
        return None

    def __copy__(self):

        # extract all attributes
        wl = self.get_blacklist_attribute_indices()
        bl = self.get_whitelist_attribute_indices()
        min_level = self._min_level
        max_level = self._max_level
        node = self.get_node()

        clone_cls = type(self.__class__.__mro__[0].__class__.__name__,
                         (self.__class__.__mro__[0], ), {'_whitelist_attribute_indices': wl,
                                                         '_blacklist_attribute_indices': bl,
                                                         '_min_level': min_level,
                                                         '_max_level': max_level})

        clone_instance: AbstractSplitRule = clone_cls(node=node)
        clone_instance.set_state(state=self.get_state().copy())

        return clone_instance

    @staticmethod
    def get_specificity() -> int:
        """
        If two splits yield the same performance,
        the split with the higher specificity will be chosen
        :return:
        """
        return 1

class AbstractNumericalSplit(AbstractSplitRule):
    """
    Represents a split over categorical attributes
    """
    def _get_attribute_candidates(self) -> typing.List[int]:
        """
        Returns a list of attribute / feature indices that Split Rule is applicable to
        """
        candidates = []
        for i, t in enumerate(self.get_tree().get_attribute_types()):
            if t == 'numerical' and self._check_rule_applicable_to_attribute_index(index=i):
                candidates.append(i)

        return candidates


class AbstractCategoricalSplit(AbstractSplitRule):
    """
    Represents a split over numerical attributes
    """

    def _get_attribute_candidates(self) -> typing.List[int]:
        """
        Returns a list of attribute / feature indices that Split Rule is applicable to
        """
        candidates = []
        attr_types = self.get_tree().get_attribute_types()
        for i, t in enumerate(attr_types):
            if t == 'categorical' and self._check_rule_applicable_to_attribute_index(index=i):
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
            for j in range(i + 1, len(supported_cols)):
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
                    rets = self._get_children_indexer_and_state(col_data_1[non_null_indexer_1_2],
                                                                col_data_2[non_null_indexer_1_2], i, j)

                    if not isinstance(rets, typing.List):
                        logging.getLogger(__package__).warning(f"Returning only one split for a split rule "
                                                               f"is deprecated"
                                                               f"but {self.__class__.mro()[0]} does."
                                                               f"if you just want to return a single split, "
                                                               f"just return [(config, [nodes...])]. Fixing "
                                                               f"that for you"
                                                               f"Code (23180198123)")
                        rets = [rets]

                    # a valid split occured?
                    for ret in rets:
                        curr_child_indices = []
                        if ret is not None:
                            null_indexer_1_2 = ~non_null_indexer_1_2
                            non_null_indices = data_indices[non_null_indexer_1_2]
                            has_null_entries = np.any(null_indexer_1_2)
                            state, assignments = ret

                            assert isinstance(state, typing.Dict), "State mus be a dict (Code: 67543256)"
                            assert isinstance(assignments, typing.Tuple), "Returned data has to be " \
                                                                          "a tuple or None (Code: 5869378963547)"
                            assert len(assignments) >= 2, "A split has to generate at least " \
                                                          "two child nodes(" \
                                                          "Code: 45698435673)"

                            # collect data indices for each node
                            has_empty_child = False
                            for child_assignment in assignments:
                                assert isinstance(child_assignment, np.ndarray) and child_assignment.dtype is np.dtype(
                                    np.bool), \
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
    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray, *args, **kwargs) -> Optional[
        typing.Tuple[typing.Dict,
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


class ThreeAttributeSplitMixin(AbstractSplitRule):
    """
    This is just a helper that deals with extracting the necessary data (including none indices)
    over exactly three attributes to easen up some repeated works
    """

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        data_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) < 3:
            return None

        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        best_state = {}

        # iterate over all attributes that are available
        # will iterate over two attribute indices where each pair is only regarded exactly once
        for i in range(len(supported_cols)):
            for j in range(i + 1, len(supported_cols)):
                for z in range(len(supported_cols)):
                    attr_idx1 = supported_cols[i]
                    attr_idx2 = supported_cols[j]
                    attr_idx3 = supported_cols[z]
                    col_data_1 = node_data[:, attr_idx1]
                    col_data_2 = node_data[:, attr_idx2]
                    col_data_3 = node_data[:, attr_idx3]
                    null_indexer_1 = self._get_col_none_indexer(column=col_data_1)
                    null_indexer_2 = self._get_col_none_indexer(column=col_data_2)
                    null_indexer_3 = self._get_col_none_indexer(column=col_data_3)
                    non_null_indexer_1 = ~null_indexer_1
                    non_null_indexer_2 = ~null_indexer_2
                    non_null_indexer_3 = ~null_indexer_3
                    non_null_indexer_1_2_3 = non_null_indexer_1 & non_null_indexer_2 & non_null_indexer_3

                    if np.any(non_null_indexer_1_2_3):
                        # get indexer for child node data assignments
                        rets = self._get_children_indexer_and_state(col_data_1[non_null_indexer_1_2_3],
                                                                    col_data_2[non_null_indexer_1_2_3],
                                                                    col_data_3[non_null_indexer_1_2_3], i,j, z
                                                                    )

                        if not isinstance(rets, typing.List):
                            logging.getLogger(__package__).warning(f"Returning only one split for a split rule "
                                                                   f"is deprecated"
                                                                   f"but {self.__class__.mro()[0]} does."
                                                                   f"if you just want to return a single split, "
                                                                   f"just return [(config, [nodes...])]. Fixing "
                                                                   f"that for you"
                                                                   f"Code (345563425)")
                            rets = [rets]

                        # a valid split occured?
                        for ret in rets:
                            curr_child_indices = []
                            if ret is not None:
                                null_indexer_1_2_3 = ~non_null_indexer_1_2_3
                                non_null_indices = data_indices[non_null_indexer_1_2_3]
                                has_null_entries = np.any(null_indexer_1_2_3)
                                state, assignments = ret

                                assert isinstance(state, typing.Dict), "State mus be a dict (Code: 345345)"
                                assert isinstance(assignments, typing.Tuple), "Returned data has to be " \
                                                                              "a tuple or None (Code: 45645645)"
                                assert len(assignments) >= 2, "A split has to generate at least " \
                                                              "two child nodes(" \
                                                              "Code: 23423423423)"

                                # collect data indices for each node
                                has_empty_child = False
                                for child_assignment in assignments:
                                    assert isinstance(child_assignment, np.ndarray) and child_assignment.dtype is np.dtype(
                                        np.bool), \
                                        "Assignments have to be a numpy array of type bool (Code: 234234056)"

                                    if not np.any(child_assignment):
                                        has_empty_child = True
                                        break

                                    curr_child_indices.append(non_null_indices[child_assignment])

                                    if has_null_entries:
                                        curr_child_indices[-1] = np.append(curr_child_indices[-1],
                                                                           data_indices[null_indexer_1_2_3])

                                # check if we have an empty child
                                # atm we do not want to accept that -> ignore
                                if not has_empty_child:
                                    child_nodes = [node.create_child_node(assigned_data_indices=indices)
                                                   for indices in curr_child_indices]

                                    state['split_attribute_indices'] = (attr_idx1, attr_idx2, attr_idx3)
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
    def _get_children_indexer_and_state(self,
                                        data_values_left: np.ndarray,
                                        data_values_middle: np.ndarray,
                                        data_values_right: np.ndarray) -> Optional[
                                            typing.List[typing.Tuple[typing.Dict,
                                                         typing.Tuple[np.ndarray]]]]:
        """

        :param data_values_left:
        :param data_values_middle:
        :param data_right:
        :return: state, tuple of child indices
        """
        pass

    def get_split_attribute_indices(self) -> typing.Tuple[int, int, int]:
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

    def get_split_attribute_names(self) -> typing.Tuple[str, str, str]:
        """
        Just a convenience method that returns the names of the split attribute
        :return:
        """
        indices = self.get_split_attribute_indices()
        return (self.get_tree().get_attribute_names()[indices[0]],
                self.get_tree().get_attribute_names()[indices[1]],
                self.get_tree().get_attribute_names()[indices[2]])


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
                rets = self._get_children_indexer_and_state(non_null_values, attr_idx)

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
                        curr_child_indices = []

                        non_null_indices = data_indices[non_null_indexer]
                        has_null_entries = np.any(null_indexer)
                        state, assignments = ret

                        assert isinstance(state, typing.Dict), "State mus be a dict (Code: 32942390)"
                        assert isinstance(assignments, typing.Tuple), "Returned data has to be " \
                                                                      "a tuple or None (Code: 00756435345)"
                        assert len(assignments) >= 2, "A split has to generate at least two child nodes (" \
                                                      "Code: 248234)"

                        # collect data indices for each node
                        has_empty_child = False
                        for child_assignment in assignments:
                            assert isinstance(child_assignment, np.ndarray) and child_assignment.dtype is np.dtype(
                                np.bool), \
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
    def _get_children_indexer_and_state(self, data_values: np.ndarray, *args, **kwargs) -> Optional[typing.List[typing.Tuple[typing.Dict,
                                                                                                            typing.Tuple[
                                                                                                                np.ndarray]]]]:
        """
        Returns for each child no
        :param data_values:
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

    def get_split_attribute_indices(self) -> typing.Tuple[int]:
        return self.get_split_attribute_index(),


class FixedValueSplit(OneAttributeSplitMixin):
    """
    Will split on discrete values (Being rather numeric with only integer values or categorical)
    on a fixed value
    """

    def get_edge_labels(self) -> typing.List[str]:
        state = self.get_state()
        value = state['value']

        return [f"is {value}", f"is NOT {value}"]

    @classmethod
    def get_help_text(cls):
        return "Will work on discrete columns. Splits the data by having a exact value,\n e.g. has_children is True."

    def user_readable_description(self) -> str:
        state = self.get_state()
        attr_name = self.get_split_attribute_name()

        if state is None:
            return "FixedValueSplit not initialized"
        else:
            val = state['value']
            return f"Split on {attr_name} equals {val}"

    def explain_split(self, sample: np.ndarray) -> str:
        state = self.get_state()
        if state is not None:
            value = state['value']
            attr_idx = self.get_split_attribute_index()
            attr_name = self.get_split_attribute_name()

            attr_value = sample[attr_idx]

            if attr_value == value:
                return f"{attr_name} matches value {value}"
            elif attr_value is None:
                return f"{attr_name} has no value available"
            else:
                return f"{attr_name} doesn't match value {value}"
        else:
            raise Exception("Fixed Value split is not initialized, hence cannot explain decision (Code: 2983742893")

    @check_initialized
    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List[int]:
        state = self.get_state()
        attr_idx = self.get_split_attribute_index()
        val = state['value']
        attribute_value = sample[attr_idx]

        if attribute_value is None:
            return [0, 1]
        elif attribute_value == val:
            return [0]
        else:
            return [1]

    def _get_attribute_candidates(self) -> typing.List[int]:
        candidates = []
        categorical_numerics = {'b',  # boolean
                                'u',  # unsigned integer
                                'i',  # signed integer
                                }
        for i, t in enumerate(self.get_tree().get_attribute_types()):
            if self._check_rule_applicable_to_attribute_index(index=i):
                if t == 'categorical':
                    candidates.append(i)
                elif t == 'numerical':
                    # check if we have some finite set of numbers
                    if self.get_tree().get_unique_values_for_attribute(i) is not None:
                        candidates.append(i)

        return candidates

    def _get_children_indexer_and_state(self, data_values: np.ndarray, index: int, *args, **kwargs):
        unique_vals = self.get_tree().get_unique_values_for_attribute(index)
        
        if unique_vals is not None:
            splits = []
    
            # check each unique val (except None)
            for val in unique_vals:
                if val is not None: # except None
                    left_childs = data_values == val
                    splits.append(({'value': val}, (left_childs, ~left_childs)))
    
            return splits

    def set_value(self, value: Union[int, str]):
        """
        Will change the split value
        :param value:
        :return:
        """
        self._state['value'] = value

    @check_initialized
    def get_fixed_val(self) -> Union[str, float, int, typing.Any]:
        return self.get_state()['value']

    def _merge(self, other, sample: Optional[np.ndarray]=None) -> Optional['AbstractSplitRule']:
        """
        :param other:
        :return:
        """

        # check if fixed val is a float
        fixed_val = self.get_fixed_val()
        is_float = False
        try:
            float(fixed_val)  # will raise if not possible
            is_float = True
        except:
            pass

        if sample is None:
            if isinstance(other, AbstractQuantileSplit):
                # we can merge some intervals iff current value is numeric AND quantile splits upper bound is >= FixedValue
                split_val_other = other.get_state()['split_value']

                if is_float:
                    if float(fixed_val) <= split_val_other:
                        # if fixed_val in range, than rule must match fixed val
                        cpy = self.__copy__()
                        cpy.set_value(fixed_val)
                        return cpy

                    else:
                        # if you ever reach that code youre trying to merge two rules that NEVER can be true together
                        # propably you dont want that
                        logging.getLogger(self.__class__.__name__).warning("You try to combine two rules which never "
                                                                           "can be True together. I will not merge them. "
                                                                           f"Nodes are {self.__class__.__name__} with state {self.get_state()} and "
                                                                           f"{other.__class__.__name__} with state {other.get_state()}."
                                                                           "Are you sure? (Code: 923487293)")
                        pass

            if isinstance(other, CloseToMedianSplit):
                # we can eat a close to median split if were inside that guy
                other_left = other.get_state()['median'] - 0.5 * other.get_state()['stdev']
                other_right = other.get_state()['median'] + 0.5 * other.get_state()['stdev']

                if is_float and other_left <= fixed_val <= other_right:
                    return self.__copy__()

            if isinstance(other, AbstractQuantileRangeSplit):
                # eat range split if we are within that range
                upper = other.get_upper_bound()
                lower = other.get_lower_bound()

                if is_float and lower <= fixed_val < upper:
                    return self.__copy__()

            if isinstance(other, FixedValueSplit):
                if fixed_val == other.get_state()['value']:
                    return self.__copy__()
                else:
                    logging.getLogger().warning("You try to merge to fixed Values having different values. "
                                                "This can never work. Do not call this function directly! "
                                                "I will ignore that. (Code: 348230948)")

        else: # we have a sample
            # We always are more specific on intervals etc. so we always return oneself
            if isinstance(other, AbstractQuantileSplit) or isinstance(other, CloseToMedianSplit) or isinstance(other, AbstractQuantileRangeSplit):
                return self.__copy__()
            elif isinstance(other, FixedValueSplit):
                sample_val = sample[self.get_split_attribute_index()]
                own_val = fixed_val
                other_val = other.get_fixed_val()

                # case I both are identical -> just eat one rule
                if own_val == other_val:
                    return FixedValueSplit.new_from_other(other=self,
                                                          state={'value': own_val})

                # case II: agrees with self (and not both are the same) -> returns elf
                elif sample_val == own_val:
                    return FixedValueSplit.new_from_other(other=self,
                                                          state={'value': own_val})
                # case III: agrees only with other
                elif sample_val == other_val:
                    return FixedValueSplit.new_from_other(other=other,
                                                          state={'value': other_val})

                # case IV: disagreeing both
                else:
                    # this may be an AND-Rule (which is not available at the moment)
                    # @TODO add an AND-Rule and connect both
                    pass

        return None

    @staticmethod
    def get_specificity() -> int:
        return 1000


@hide_in_ui
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
                    return f"{attr_name} is close to groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ = {0.5 * round(stddev, 2)})"
                else:
                    return f"{attr_name} is outside of groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ = {0.5 * round(stddev, 2)})"
        else:
            raise Exception("Close To Median Split not initialized, hence, "
                            "cannot explain a decision (Code: 234678234902347)")

    @check_initialized
    def get_lower_bound(self):
        state = self.get_state()
        median = state['median']
        stdev = state['stdev']

        return median - 0.5 * stdev

    @check_initialized
    def get_upper_bound(self):
        state = self.get_state()
        median = state['median']
        stdev = state['stdev']

        return median + 0.5 * stdev

    def get_edge_labels(self):
        return ["close to median val", "outside median val"]

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
                   f"{median} (± ½ × σ = {0.5 * round(stdev, 2)})"
        else:
            return "Close To Median Split is not initialized"

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        state = self.get_state()
        if state is not None:
            attr_idx = self.get_split_attribute_index()
            median = state['median']
            stdev = state['stdev']
            val = sample[attr_idx]

            if val is None:
                return [0, 1]
            else:
                if abs(median - val) <= 0.5 * stdev:
                    return [0]
                else:
                    return [1]

    def _get_children_indexer_and_state(self, data_values: np.ndarray, *args, **kwargs):

        # get median val and standard deviation
        median_val = np.median(data_values)
        stdev = np.std(data_values)

        # everything close to median by means of stddev goes to left node
        inside_median = np.abs(data_values - median_val) <= 0.5 * stdev

        state = {'median': median_val, 'stdev': stdev}

        return [(state, (inside_median, ~inside_median))]

    def _merge(self, other, sample: Optional[np.ndarray]=None) -> Optional['AbstractSplitRule']:
        self_state = self.get_state()
        other_state = other.get_state()
        self_left = self_state['median'] - 0.5 * self_state['stdev']
        self_right = self_state['median'] + 0.5 * self_state['stdev']

        if isinstance(other, CloseToMedianSplit):
            other_left = other_state['median'] - 0.5 * other_state['stdev']
            other_right = other_state['median'] + 0.5 * other_state['stdev']

            # we can eat a median split if it completely covers us
            if other_left <= self_left and other_right >= self_right:
                return self.__copy__()

            # I overlap right (left version is equivalent and will be checked by switched operators)
            if other_left < self_left < other_right < self_right:
                new_left = self_left
                new_right = other_right
                cpy = self.__copy__()
                cpy._state['median'] = (new_left + new_right) / 2
                cpy._state['stdev'] = (new_right - new_left)

                return cpy

        if isinstance(other, AbstractQuantileSplit):
            # if we are below other threshold, the var must lay inside us
            # so we consume the QuantileSplit completely
            other_right = other.get_state()['split_value']
            if self_right <= other_right:
                return self.__copy__()

            # check if we hang out right
            if self_left < other_right < self_right:
                new_median = (self_left + other_right) / 2
                new_stdev = (other_right - self_left) / 2
                cpy = self.__copy__()
                cpy._state['stdev'] = new_stdev
                cpy._state['median'] = new_median

                return cpy


        return None


class AbstractQuantileRangeSplit(AbstractNumericalSplit, OneAttributeSplitMixin):
    """
    Will divide the data into Quantiles and splits data laying inside a span of these or outside
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @check_initialized
    def get_lower_bound(self) -> float:
        return self.get_state()['lower_bound']

    @check_initialized
    def get_upper_bound(self) -> float:
        return self.get_state()['upper_bound']

    @classmethod
    def get_help_text(cls):
        return "Will divide the data into given amount of uniorm quantiles like:\n" \
               "[Q1|Q2|Q3|Q4|Q5|....]. all pairs of interval boundaries are than evaluated to split the data" \
               "being inside the interval or outside,\n e.g. Age is between 20 and 40"

    def explain_split(self, sample: np.ndarray):

        if self.is_initialized():
            attr_index = self.get_split_attribute_index()
            attr_name = self.get_split_attribute_name()
            val = sample[attr_index]
            lower = self.get_lower_bound()
            upper = self.get_upper_bound()

            if val is None:
                return f"alue for {attr_name} is not available, hence assigned to all children"
            else:
                if lower <= val < upper:
                    return f"{attr_name} is INSIDE range [{lower:.2f}, ... {upper:.2f}["
                else:
                    if val < lower:
                        way = 'below'
                    else:
                        way = 'above'
                    return f"{attr_name} is OUTSIDE range [{lower:.2f}, ... {upper:.2f}[ ({val:.2f} is {way} range)"
        else:
            raise Exception("Close To Median Split not initialized, hence, "
                            "cannot explain a decision (Code: 234723948723498237)")

    def get_edge_labels(self):
        return ["inside interval", "outside Intervall"]

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        if self.is_initialized():
            attr_name = self.get_split_attribute_name()
            lower = self.get_lower_bound()
            upper = self.get_upper_bound()

            return f"{attr_name} is within range [{lower:.2f}, ..., {upper:.2f}["
        else:
            return "Quantile range split not initialized"

    @check_initialized
    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        attr_idx = self.get_split_attribute_index()
        lower = self.get_lower_bound()
        upper = self.get_upper_bound()
        val = sample[attr_idx]

        if val is None:
            return [0, 1]
        else:
            if lower <= val < upper:
                return [0]
            else:
                return [1]

    @property
    @abstractmethod
    def quantile_count(self) -> int:
        """
        Gets the amount of bins that should be created
        :return:
        """
        pass

    @check_initialized
    def is_inside(self, sample: np.ndarray) -> bool:
        """
        Checks if the given sample falls inside the interval
        :param sample:
        :return:
        """
        lower = self.get_lower_bound()
        upper = self.get_upper_bound()
        val = sample[self.get_split_attribute_index()]


        return lower <= val < upper

    def _get_children_indexer_and_state(self, data_values: np.ndarray, *args, **kwargs):

        quantiles = self.quantile_count
        assert isinstance(quantiles, int) and quantiles >= 1, "Quantiles must be integers >= 1 (Code: 2342323)"

        quantile_parts = [(q + 1) / (quantiles + 1) for q in range(quantiles + 1)]
        quantile_vals = np.quantile(data_values, quantile_parts)
        quantile_vals = list(set(list(quantile_vals))) # this will remove doubled entries
        results = []

        for left_q in range(len(quantile_parts) - 1):
            for right_q in range(left_q + 1, len(quantile_vals)):
                left = (data_values >= quantile_vals[left_q]) &  (data_values < quantile_vals[right_q])
                results.append(({'lower_bound': quantile_vals[left_q],
                                 'upper_bound': quantile_vals[right_q]},
                                 (left, ~left)))

        return results

    def _merge(self, other, sample: Optional[np.ndarray]=None) -> Optional['AbstractSplitRule']:
        # consume other range split if it is embedding us completely

        if sample is None: # case: No sample given
            if isinstance(other, AbstractQuantileRangeSplit):
                if self.get_lower_bound() >= other.get_lower_bound() and \
                        self.get_upper_bound() <= other.get_upper_bound():
                    return self.__copy__()

            # consume other close to median split if we are completely embedded within
            if isinstance(other, CloseToMedianSplit):
                if self.get_lower_bound() >= other.get_lower_bound() and \
                        self.get_upper_bound() <= other.get_upper_bound():
                    return self.__copy__()

            # consume Quantile splits if upper bound is below our upper bound
            if isinstance(other, AbstractQuantileSplit):
                if self.get_upper_bound() <= other.get_upper_bound():
                    return self.__copy__()

        else: # case: Merge according to sample
            sample_val = sample[self.get_split_attribute_index()]

            if sample_val is None: # no value so any rule is fine
                return self.__copy__()

            if isinstance(other, AbstractQuantileRangeSplit):
                lower_starting_rule = self
                higher_starting_rule = other
                if self.get_lower_bound() > other.get_lower_bound():
                    lower_starting_rule = other
                    higher_starting_rule = self

                higher_ending_rule = self
                lower_ending_rule = other

                if self.get_upper_bound() < other.get_upper_bound():
                    higher_ending_rule = other
                    lower_ending_rule = self

                lower_starting_val = lower_starting_rule.get_lower_bound()
                higher_starting_val = higher_starting_rule.get_lower_bound()


                lower_ending_val = lower_ending_rule.get_upper_bound()
                higher_ending_val = higher_ending_rule.get_upper_bound()

                in_lower = lower_starting_rule.is_inside(sample=sample)
                in_upper = higher_starting_rule.is_inside(sample=sample)

                if sample_val < lower_starting_val:  # case IV: Below both
                    q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                    'split_value': lower_starting_val})
                    return q_split

                elif sample_val >= higher_ending_val:  # case III: above both
                    q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0, 
                                                                                    'split_value': higher_ending_val})
                    return q_split

                elif in_lower and in_upper: # case I: inside both
                    qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                          'upper_bound': lower_ending_val,
                                                                                          'lower_bound': higher_starting_val})
                    return qr_split

                elif in_lower and not in_upper:  # case II: within lower but not upper
                    qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                          'upper_bound': higher_starting_rule.get_lower_bound(),
                                                                                          'lower_bound': lower_starting_rule.get_lower_bound()})
                    return qr_split

                elif in_upper and not in_lower:  # case V: within upper but on in lower
                    qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                          'upper_bound': higher_starting_rule.get_upper_bound(),
                                                                                          'lower_bound': lower_starting_rule.get_upper_bound()})
                    return qr_split

                elif not in_upper and not in_lower and lower_ending_val <= sample_val < higher_starting_val:
                    # case VI rules do not overlap and sample is between both
                    qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                          'upper_bound': higher_starting_rule.get_lower_bound(),
                                                                                          'lower_bound': lower_starting_rule.get_upper_bound()})
                    return qr_split

            elif isinstance(other, AbstractQuantileSplit):
                upper_val = self.get_upper_bound()
                lower_val = self.get_lower_bound()
                split_val = other.get_split_value()

                if lower_val <= split_val < upper_val:
                    # case cat I (splits overlap)

                    # case I.I sample below
                    if sample_val < lower_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': lower_val})
                        return q_split

                    # case I.II sample inside
                    if lower_val <= sample_val < split_val:
                        qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                              'upper_bound': split_val,
                                                                                              'lower_bound': lower_val})
                        return qr_split

                    # case I.III sample in range but aboove quantile split
                    if split_val <= sample_val < upper_val:
                        qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                              'upper_bound': upper_val,
                                                                                              'lower_bound': split_val})
                        return qr_split

                    # case I.IV Sample above all
                    if sample_val >= upper_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': upper_val})
                        return q_split

                elif split_val < lower_val:  #  case cat II: quantile split split below
                    # II.1 split val below everything
                    if sample_val < split_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': split_val})
                        return q_split

                    # II.II ... between
                    if split_val <= sample_val < lower_val:
                        qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                              'upper_bound': lower_val,
                                                                                              'lower_bound': split_val})
                        return qr_split

                    # II.III is in range
                    if self.is_inside(sample):
                        return self.__copy__()
                    
                    # II.IV above all
                    if sample_val >= upper_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': upper_val})
                        return q_split

                elif upper_val < split_val:
                    # Cat III rannge split below

                    # III.I sample below all
                    if sample_val < lower_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': lower_val})
                        return q_split
                    
                    # III.II sample inside range
                    if self.is_inside(sample):
                        return self.__copy__()

                    # III.III sample is above range but below quantile
                    if upper_val <= sample_val < split_val:
                        qr_split = MergedQuantileRangeSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                              'upper_bound': split_val,
                                                                                              'lower_bound': upper_val})
                        return qr_split

                    # III.IV above all
                    if split_val < sample_val:
                        q_split = MergedQuantileSplit.new_from_other(other=self, state={'quantile': 0,
                                                                                        'split_value': split_val})
                        return q_split


@hide_in_ui
class MergedQuantileRangeSplit(AbstractQuantileRangeSplit):
    """
    This split will only be generated by merging different rules
    :param AbstractQuantileRangeSplit:
    :return:
    """
    @property
    def quantile_count(self) -> int:
        return 0


class TenQuantileRangeSplit(AbstractQuantileRangeSplit):
    @property
    def quantile_count(self) -> int:
        return 9


class TwentyQuantileRangeSplit(AbstractQuantileRangeSplit):
    @property
    def quantile_count(self) -> int:
        return 19


@hide_in_ui
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

    def get_edge_labels(self):
        attr_name_1, attr_name_2 = self.get_split_attribute_names()
        return [f"{attr_name_1} < {attr_name_2}", f"{attr_name_1} ≥ {attr_name_2}"]

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available, hence assigned to both children"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available, hence assigned to both children"
            else:
                if attr1 < attr2:
                    return f"{attr_name_1} < {attr_name_2}"
                else:
                    return f"{attr_name_1} ≥ {attr_name_2}"

        else:
            raise Exception("Smaller Than Split not initialized, hence, "
                            "cannot explain a decision (Code: 234237423987423)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]

        if attr1 is None or attr2 is None:
            return [0, 1]
        else:
            if attr1 < attr2:
                return [0]
            else:
                return [1]

    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray, *args, **kwargs):
        left_vals = data_values_left < data_values_right
        state = {}

        return [(state, (left_vals, ~left_vals))]

@hide_in_ui
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

    def get_edge_labels(self):
        attr_name_1, attr_name_2 = self.get_split_attribute_names()
        return [f"{attr_name_1} < ½ × {attr_name_2}", f"{attr_name_1} ≥ ½ × {attr_name_2}"]

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available, hence assigned to both children"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available, hence assigned to both children"
            else:
                if attr1 < 0.5 * attr2:
                    return f"{attr_name_1} < ½ × {attr_name_2}"
                else:
                    return f"{attr_name_1} ≥ ½ × {attr_name_2}"

        else:
            raise Exception("Less Than Half Of Split not initialized, hence, "
                            "cannot explain a decision (Code: 23423234234234)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]

        if attr1 is None or attr2 is None:
            return [0, 1]
        else:
            if attr1 < 0.5 * attr2:
                return [0]
            else:
                return [1]

    def _get_children_indexer_and_state(self,
                                        data_values_left: np.ndarray,
                                        data_values_right: np.ndarray, *args, **kwargs):
        left_vals = data_values_left < 0.5 * data_values_right
        state = {}

        return [(state, (left_vals, ~left_vals))]


@hide_in_ui
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
            return f"Split on categorical attribute {attr_name} with possible values: {', '.join([*lookup.keys()])}"
        else:
            return "Single Category Split is not evaluated"

    def get_edge_labels(self):
        state = self.get_state()
        lookup = state['label_to_node_idx_lookup']
        reverse_dict = {value: key for key, value in lookup.items()}
        labels = []

        for i in range(len(lookup)):
            labels.append(reverse_dict[i])

        return labels

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx = self.get_split_attribute_index()
            attr_name = self.get_split_attribute_name()
            attr = sample[attr_idx]
            lookup = state['label_to_node_idx_lookup']

            if attr is None:
                return f"{attr_name} is not available, hence assigned to all child nodes"
            elif attr not in lookup:  # we did not split on that specific val
                return f"{attr_name} with value {attr} was not available at that stage during training, " \
                       f"hence assigned to all childs"
            else:
                attr_node = lookup[attr]
                return f"{attr_name} has value {attr}, hence assigned it to node number {attr_node+1}"

        else:
            raise Exception("Single Category Split Split not initialized, hence, "
                            "cannot explain a decision (Code: 365345673459683456)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        state = self.get_state()
        lookup = state['label_to_node_idx_lookup']
        attr_idx = self.get_split_attribute_index()
        val = sample[attr_idx]

        # return all nodes if we encounter an invalid value´(at training time) or a None
        if val is None or val not in lookup:
            return [*range(len(lookup))]
        else:
            return [lookup[val]]

    def _get_children_indexer_and_state(self, data_values: np.ndarray, *args, **kwargs):

        distinct_node_labels = np.unique(data_values)
        node_indexer = []
        label_to_node_idx_lookup = {}

        # only split if we have at least two distinct values
        if len(distinct_node_labels) >= 2:
            for i, lbl in enumerate(distinct_node_labels):
                node_indexer.append(data_values == lbl)
                label_to_node_idx_lookup[lbl] = i

            return [({'label_to_node_idx_lookup': label_to_node_idx_lookup},
                     tuple(node_indexer))]

        return None


class AbstractQuantileSplit(AbstractNumericalSplit, OneAttributeSplitMixin):
    """
    Will to split like a NumericalSplit but only over #quantile threasholds
    implement specific version with quantile_count set
    """
    @property
    @abstractmethod
    def quantile_count(self) -> int:
        """
        Returns the amount of quantiles you want to check
        :return:
        """
        pass

    @classmethod
    def get_help_text(cls):
        return "Will divide the data into a given number of uniform quantiles at each attribute.\n" \
               "Example: [Q1|Q2|Q3|Q4|...]. \n" \
               "Each separator (|) will be tried as split value to divde data in above and below," \
               "e.g. age < 30"

    @check_initialized
    def get_upper_bound(self) -> float:
        """
        Will return the split value
        :return:
        """
        return self.get_state()['split_value']

    @check_initialized
    def get_split_value(self) -> float:
        return self.get_upper_bound()

    @check_initialized
    def get_quantile(self) -> int:
        return self.get_state()['quantile']


    @check_initialized
    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        attr_name = self.get_split_attribute_name()
        split_val = state['split_value']
        attr_index = self.get_split_attribute_index()

        attr = sample[attr_index]
        quantile_val = state['quantile']
        quantile_str = f"{quantile_val + 1}/{self.quantile_count + 1}"
        if attr is None:
            return f"Attribute {attr_name} is not available"
        else:
            if attr < split_val:
                return f"{attr_name} < " \
                       f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"
            else:
                return f"{attr_name} ≥ " \
                       f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"

    def get_edge_labels(self):
        state = self.get_state()
        split_val = state['split_value']
        return [f"< {round(split_val, 2)}", f"≥  {round(split_val, 2)}"]

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()

        if state is not None:
            attr_name = self.get_split_attribute_name()
            split_val = state['split_value']
            quantile_val = state['quantile']

            if quantile_val != 0:
                quantile_str = f"{quantile_val + 1}/{self.quantile_count + 1}"
                return f"{attr_name} < " \
                       f"{round(split_val, 2) } (=Groups' {quantile_str}. Quantile)"
            else:
                return f"{attr_name} < " \
                       f"{round(split_val, 2)}"


        else:
            return "Quantile Split split not initialized"

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        state = self.get_state()
        val = state['split_value']
        attr_idx = self.get_split_attribute_index()

        attr = sample[attr_idx]
        if attr is None:
            return [0, 1]
        else:
            if attr < val:
                return [0]
            else:
                return [1]

    def _get_children_indexer_and_state(self, data_values: np.ndarray, *args, **kwargs):
        quantiles = self.quantile_count
        assert isinstance(quantiles, int) and quantiles >= 1, "Quantiles must be integers >= 1 (Code: 3284723894)"

        quantile_parts = [(q + 1) / (quantiles + 1) for q in range(quantiles)]
        quantile_vals = np.quantile(data_values, quantile_parts)
        results = []
        for i, q_val in enumerate(quantile_vals):
            left = data_values < q_val
            results.append(
                ({'split_value': q_val,
                  'quantile': i}, (left, ~left))
            )

        return results

    def _merge(self, other, sample: Optional[np.ndarray]=None):
        if sample is None:
            if isinstance(other, AbstractQuantileSplit) and self.quantile_count >= other.quantile_count:
                state_own = self.get_state()
                state_other = other.get_state()

                if sample is None:  # case: no sample mode
                    cpy = self.__copy__()
                    state_new = state_own.copy()
                    state_new['split_value'] = min(state_own['split_value'], state_other['split_value'])
                    state_new['quantile'] = min(state_own['quantile'], state_other['quantile'])
                    cpy.set_state(state_new)

                    return cpy
                else:
                    attr_idx = self.get_split_attribute_index()
                    sample_val = sample[attr_idx]

                    # if sample val is None we just return a single rule
                    if sample_val is None:
                        return self.__copy__()

                    upper_rule = self if self.get_split_value() > other.get_split_value() else other
                    lower_rule = self if self.get_split_value() <= other.get_split_value() else other
                    upper_val = upper_rule.get_split_value()
                    lower_val = lower_rule.get_split_value()

                    if sample_val < lower_val:  # case: below the lower rule
                        return lower_rule.__copy__()
                    elif sample_val < upper_val: # between the rules
                        state = {'lower_bound': lower_val, 'upper_bound': upper_val}
                        range = MergedQuantileRangeSplit.new_from_other(other=self, state=state)

                        return range
                    else:  #case: above the highest
                        return upper_rule.__copy__()
        else:
            if isinstance(other, AbstractQuantileSplit):
                sample_val = sample[self.get_split_attribute_index()]
                upper = self if self.get_split_value() > other.get_split_value() else other
                lower = self if self.get_split_value() <= other.get_split_value() else other

                lower_val = lower.get_split_value()
                upper_val = upper.get_split_value()

                # case I below both
                if sample_val < lower_val:
                    return lower.__copy__()

                # case II: Between both -> range
                if lower_val <= sample_val < upper_val:
                    return MergedQuantileRangeSplit.new_from_other(other=self, state={'upper_bound': upper_val,
                                                                               'lower_bound': lower_val})

                # case III
                if sample_val >= upper_val:
                    return upper.__copy__()



        return None

@hide_in_ui
class MergedQuantileSplit(AbstractQuantileSplit):
    @property
    def quantile_count(self) -> int:
        return 0

class FiveQuantileSplit(AbstractQuantileSplit):
    @property
    def quantile_count(self):
        return 4


class TenQuantileSplit(AbstractQuantileSplit):

    @property
    def quantile_count(self):
        return 9


class TwentyQuantileSplit(AbstractQuantileSplit):

    @property
    def quantile_count(self):
        return 19

@hide_in_ui
class MedianSplit(AbstractQuantileSplit):
    """
    Will separate on Median Element
    """

    @property
    def quantile_count(self):
        return 1

@hide_in_ui
class OneQuantileSplit(MedianSplit):
    """
    Just a convinient name
    """
    pass


class AbstractMultiplicativeQuantileSplit(TwoAttributeSplitMixin, AbstractNumericalSplit):
    """
    Will multiply two attributes and finds best cut point dividing the multiplicative result into
    given amount of quantiles
    """

    @property
    @abstractmethod
    def quantile_count(self) -> int:
        """
        How many quantiles the data should be divided to?
        :return:
        """
        pass

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
            split_val = state['split_value']

            quantile_val = state['quantile']
            quantile_str = f"{quantile_val+1}/{self.quantile_count+1}"

            return f"{attr_name_1} * {attr_name_2} < " \
                   f"{round(split_val, 2) } (=Groups' {quantile_str}. Quantile)"
        else:
            return "Multiplicative split is not initialized"

    def get_edge_labels(self):
        state = self.get_state()
        split_val = state['split_value']
        return [f"< {round(split_val, 2)}", f"≥  {round(split_val, 2)}"]

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available"
            else:
                split_val = state['split_value']
                quantile_val = state['quantile']
                quantile_str = f"{quantile_val+1}/{self.quantile_count+1}"

                if (attr1 * attr2) < split_val:
                    return f"{attr_name_1} * < {attr_name_2} <" \
                           f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"
                else:
                    return f"{attr_name_1} * < {attr_name_2} ≥ " \
                           f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"
        else:
            raise Exception("Multiplicative Split is not initialized, hence "
                            "cannot explain a decision (Code: 345348752376)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]
        split_val = self.get_state()['split_value']
        if attr1 is None or attr2 is None:
            return [0, 1]
        else:
            if (attr1 * attr2) < split_val:
                return [0]
            else:
                return [1]

    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray, *args, **kwargs):
        quantiles = self.quantile_count
        assert isinstance(quantiles, int) and quantiles >= 1, "Quantiles must be integers >= 1 (Code: 45645645)"

        multiplicative_feature = data_values_left * data_values_right

        quantile_parts = [(q + 1) / (quantiles + 1) for q in range(quantiles)]
        quantile_vals = np.quantile(multiplicative_feature, quantile_parts)

        results = []
        for i, q_val in enumerate(quantile_vals):
            left = multiplicative_feature < q_val
            results.append(
                ({'split_value': q_val,
                  'quantile': i}, (left, ~left))
            )

        return results

@hide_in_ui
class MedianMultiplicativeQuantileSplit(AbstractMultiplicativeQuantileSplit):
    @property
    def quantile_count(self):
        return 1

@hide_in_ui
class TenQuantileMultiplicativeSplit(AbstractMultiplicativeQuantileSplit):
    @property
    def quantile_count(self):
        return 9

@hide_in_ui
class TwentyQuantileMultiplicativeSplit(AbstractMultiplicativeQuantileSplit):
    @property
    def quantile_count(self):
        return 19

@hide_in_ui
class FiveQuantileMultiplicativeSplit(AbstractMultiplicativeQuantileSplit):
    @property
    def quantile_count(self):
        return 4


class AbstractAdditiveQuantileSplit(TwoAttributeSplitMixin, AbstractNumericalSplit):
    """
    Will add two attributes and findsbest cut point dividing the additive result into
    given amount of quantiles
    """

    @property
    @abstractmethod
    def quantile_count(self) -> int:
        """
        How many quantiles the data should be divided to?
        :return:
        """
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attributes: typing.Tuple[int, int] = None

    def get_edge_labels(self):
        state = self.get_state()
        split_val = state['split_value']
        return [f"< {round(split_val, 2)}", f"≥  {round(split_val, 2)}"]

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        state = self.get_state()
        if state is not None:
            attr_name_1, attr_name_2 = self.get_split_attribute_names()
            split_val = state['split_value']

            quantile_val = state['quantile']
            quantile_str = f"{quantile_val+1}/{self.quantile_count+1}"

            return f"{attr_name_1} + {attr_name_2} < " \
                   f"{round(split_val, 2) } (=Groups' {quantile_str}. Quantile)"
        else:
            return "Multiplicative split is not initialized"

    def explain_split(self, sample: np.ndarray):
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]

            attr_name_1, attr_name_2 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available"
            else:
                split_val = state['split_value']
                quantile_val = state['quantile']
                quantile_str = f"{quantile_val+1}/{self.quantile_count+1}"

                if (attr1 * attr2) < split_val:
                    return f"{attr_name_1} + < {attr_name_1} <" \
                           f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"
                else:
                    return f"{attr_name_1} + < {attr_name_2} ≥ " \
                           f"{round(split_val, 2)} (=Groups' {quantile_str}. Quantile)"
        else:
            raise Exception("Additive Split is not initialized, hence "
                            "cannot explain a decision (Code: 3453454)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_node_indices_for_sample(sample=sample)
        attr_idx_1, attr_idx_2 = self.get_split_attribute_indices()
        attr1 = sample[attr_idx_1]
        attr2 = sample[attr_idx_2]
        split_val = self.get_state()['split_value']
        if attr1 is None or attr2 is None:
            return [0, 1]
        else:
            if (attr1 + attr2) < split_val:
                return [0]
            else:
                return [1]

    def _get_children_indexer_and_state(self, data_values_left: np.ndarray, data_values_right: np.ndarray, *args, **kwargs):
        quantiles = self.quantile_count
        assert isinstance(quantiles, int) and quantiles >= 1, "Quantiles must be integers >= 1 (Code: 34534534)"

        multiplicative_feature = data_values_left + data_values_right

        quantile_parts = [(q + 1) / (quantiles + 1) for q in range(quantiles)]
        quantile_vals = np.quantile(multiplicative_feature, quantile_parts)

        results = []
        for i, q_val in enumerate(quantile_vals):
            left = multiplicative_feature < q_val
            results.append(
                ({'split_value': q_val,
                  'quantile': i}, (left, ~left))
            )

        return results

@hide_in_ui
class MedianAdditiveQuantileSplit(AbstractAdditiveQuantileSplit):
    @property
    def quantile_count(self):
        return 1

@hide_in_ui
class TenQuantileAdditiveSplit(AbstractAdditiveQuantileSplit):
    @property
    def quantile_count(self):
        return 9

@hide_in_ui
class TwentyQuantileAdditiveSplit(AbstractAdditiveQuantileSplit):
    @property
    def quantile_count(self):
        return 19

@hide_in_ui
class FiveQuantileAdditiveSplit(AbstractAdditiveQuantileSplit):
    @property
    def quantile_count(self):
        return 4

@hide_in_ui
class FixedChainRule(AbstractSplitRule):
    """
    Represents a ruleset that is a list of paramterized split rules
    that will be connected using AND
    """
    _rules_and_expected_indices: typing.List[typing.Tuple[AbstractSplitRule, int]] = None
    _name: str = None
    _version: float = 0.02
    _attribute_names: typing.List = []

    def get_edge_labels(self) -> typing.List[str]:
        return ['yes', 'no']

    def __init__(self, *args, **kwargs):
        node = kwargs['node']
        # change state of the rule to match preconditions
        # print(node.get_tree().get_attribute_names(), self._attribute_names)
        if node.get_tree().get_attribute_names() != self._attribute_names:
            raise Exception(f"Attribute names of extracted rule ({self._attribute_names}) does not match to "
                            f"current trees atrributes ({node.get_tree().get_attribute_names()})")

        for rule, indices in self._rules_and_expected_indices:
            rule._node = node
            rule._child_nodes = [None, None]
            rule._assigned_data_indices = [None, None]
            rule._is_evaluated = True

        super().__init__(*args, **kwargs)
        self._state = {
            '_rules_and_expected_indices': self._rules_and_expected_indices,
            '_name': self._name,
            '_version': self._version,
            '_attribute_names': self._attribute_names
        }

    def user_readable_description(self) -> str:
        """
        Will print all single rules
        :return:
        """
        rules_str_array = [str(rule[0].user_readable_description()) for rule in self._rules_and_expected_indices]
        rules_str = ''
        for i, rule_str in enumerate(rules_str_array):
            rules_str += f'\n  ({i+1}) {rule_str}'
        return f"Chain rule \'{self._name}\' consisting of {len(self._rules_and_expected_indices)} steps:" + rules_str

    def explain_split(self, sample: np.ndarray) -> str:
        res = self._execute(samples=sample.reshape((1, -1)))[0]
        if res:
            return f"Chain rule \ \'{self._name}\' did pass"
        else:
            return f"Chain rule \ \'{self._name}\' did NOT pass"

    def get_child_node_indices_for_sample(self, sample: np.ndarray):
        if self._execute(sample.reshape((1, -1)))[0]:
            return [0]
        else:
            return [1]

    def _get_best_split(self) -> typing.Optional[typing.Tuple[float, typing.List['Node']]]:
        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()
        data_indexer = self._execute(node_data)

        if np.any(data_indexer) and np.any(~data_indexer):
            child_nodes = [node.create_child_node(node_indices[data_indexer]),
                           node.create_child_node(node_indices[~data_indexer])]
            self.set_child_nodes(child_nodes=child_nodes)
            # self.set_child_nodes(child_nodes=child_nodes)
            self.set_state({})
            score = self.get_information_measure()(parent_node=self.get_node())

            return score, child_nodes

        return None

    def get_name(self) -> str:
        return self._name

    def _execute(self, samples: np.ndarray) -> np.array:
        """
        Applies the rules onto each sample
        returning True when they match ALL rules otherwise False

        :param samples:
        :return:
        """
        ret = np.ndarray(shape=(len(samples),), dtype=np.bool)
        for i, sample in enumerate(samples):
            accepts: bool = True
            for rule_dummy, expected_index in self._rules_and_expected_indices:
                indices = rule_dummy.get_child_node_indices_for_sample(sample=sample)
                if expected_index not in indices:
                    accepts = False
                    break  # we can stop here since we are not interested for the other results

            ret[i] = accepts

        return ret

    def _get_attribute_candidates(self) -> typing.List[int]:
        pass

    @classmethod
    def save_to_file(cls, out_file: str):
        """
        Saves the rule to the disk
        :param out_file:
        :return:
        """
        with open(out_file, 'w') as f:
            # this will pickle the rule name (class name) and the _state attribute
            # together with the expected indices each
            data = cls._rules_and_expected_indices
            d = {'rules_and_expected_indices': [(rule.__class__.__name__, rule._state, expected_indices) for rule, expected_indices in data],
                 'name': cls._name,
                 'version': cls._version,
                 'attribute_names': cls._attribute_names}
            jd = json.dumps(d)
            f.write(jd)

    @classmethod
    def load_from_file(cls, in_file: str):
        """
        Loads the rule from disk
        :param in_file:
        :return:
        """
        with open(in_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or data['version'] != cls._version:
            raise Exception(f"Could not load file since version does not match")

        rules_and_expected_indices = []
        for rule, config, expected_indices in data['rules_and_expected_indices']:
            if rule in globals():
                # load the class
                kls = globals()[rule](node=None)
                # set config
                kls._state = config
                # create list with intanciated and setup class
                rules_and_expected_indices.append((kls, expected_indices))
            else:
                raise Exception("I tried to instantiate rule {rule} but could not find it"
                                " in this package (Code: 88999)")
        return type(cls.__mro__[0].__name__, (cls,), {'_name': data['name'],
                                                      '_rules_and_expected_indices': rules_and_expected_indices,
                                                      '_attribute_names': data['attribute_names']})

    @classmethod
    def from_node(cls, target_node: 'Node', name: str) -> typing.Type:
        from hdtree.hd_tree_classes.node import Node

        split_rules: typing.List[typing.Tuple[AbstractSplitRule, int]] = []
        while target_node.get_parent():
            # extract rule path (backwards) get states and child node order
            node: Node = target_node
            parent = node.get_parent()
            parent_rule = parent.get_split_rule()
            parent_rule_state = parent_rule.get_state()
            parent_rule_class = parent_rule.__class__
            expected_child_number = parent_rule.get_child_nodes().index(node)

            parent_rule_dummy = parent_rule_class(node=None)
            parent_rule_dummy._state = parent_rule_state

            split_rules.append((parent_rule_dummy, expected_child_number))
            target_node = parent

        split_rules.reverse()

        # generate a dummy class (no instance!) that behaves like a normal Split Rule thingy
        rule = type(cls.__mro__[0].__name__, (cls,), {'_name': name,
                                                      '_rules_and_expected_indices': split_rules,
                                                      '_is_evaluated': False,
                                                      '_attribute_names': target_node.get_tree().get_attribute_names(),
                                                      })

        return rule

    def get_split_attribute_indices(self) -> typing.Tuple[int]:
        indices = []
        for rule_dummy, expected_index in self._rules_and_expected_indices:
            indices += rule_dummy.get_split_attribute_indices()

        return list(set(indices))


class MultiplicativeSmallerThanSplit(ThreeAttributeSplitMixin, AbstractNumericalSplit):
    """
    Splits on attribute1 * attribute2 < attribute3
    """
    def get_edge_labels(self) -> typing.List[str]:
        attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()

        return [f"{attr_name_1} * {attr_name_2} < {attr_name_3}",
                f"{attr_name_1} * {attr_name_2} ≥ {attr_name_3}"]

    def user_readable_description(self) -> str:
        state = self.get_state()
        if state is not None:
            attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()
            return f"{attr_name_1} * {attr_name_2} < " \
                   f"{attr_name_3}"
        else:
            return "Multiplicative smaller than split is not initialized"

    def explain_split(self, sample: np.ndarray) -> str:
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2, attr_idx_3 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]
            attr3 = sample[attr_idx_3]

            attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available"
            elif attr3 is None:
                return f"Attribute {attr_name_3} is not available"
            else:
                if attr_idx_1 * attr_idx_2 < attr_idx_3:
                    return f"{attr_name_1} * {attr_name_2} < {attr_name_3}"
                else:
                    return f"{attr_name_1} * {attr_name_2} ≥ {attr_name_3}"

        else:
            raise Exception("Multiplicative smaller than split is not initialized, hence "
                            "cannot explain a decision (Code: 3453454)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List[int]:
        attr_idx_1, attr_idx_2, attr_idx_3 = self.get_split_attribute_indices()
        if sample[attr_idx_1] * sample[attr_idx_2] < attr_idx_3:
            return [0]
        else:
            return [1]

    def _get_children_indexer_and_state(self,
                                        data_values_left: np.ndarray,
                                        data_values_middle: np.ndarray,
                                        data_values_right: np.ndarray, *args, **kwargs) -> Optional[
                                            typing.List[typing.Tuple[typing.Dict,
                                                         typing.Tuple[np.ndarray]]]]:

        left_indices = (data_values_left * data_values_left) < data_values_right
        return [({}, (left_indices, ~left_indices))]


class AdditiveSmallerThanSplit(ThreeAttributeSplitMixin, AbstractNumericalSplit):
    """
    Splits on attribute1 + attribute2 < attribute3
    """
    def get_edge_labels(self) -> typing.List[str]:
        attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()

        return [f"{attr_name_1} + {attr_name_2} < {attr_name_3}",
                f"{attr_name_1} + {attr_name_2} ≥ {attr_name_3}"]

    def user_readable_description(self) -> str:
        state = self.get_state()
        if state is not None:
            attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()
            return f"{attr_name_1} + {attr_name_2} < " \
                   f"{attr_name_3}"
        else:
            return "Multiplicative smaller than split is not initialized"

    def explain_split(self, sample: np.ndarray) -> str:
        state = self.get_state()
        if state is not None:
            attr_idx_1, attr_idx_2, attr_idx_3 = self.get_split_attribute_indices()
            attr1 = sample[attr_idx_1]
            attr2 = sample[attr_idx_2]
            attr3 = sample[attr_idx_3]

            attr_name_1, attr_name_2, attr_name_3 = self.get_split_attribute_names()

            if attr1 is None:
                return f"Attribute {attr_name_1} is not available"
            elif attr2 is None:
                return f"Attribute {attr_name_2} is not available"
            elif attr3 is None:
                return f"Attribute {attr_name_3} is not available"
            else:
                if attr_idx_1 * attr_idx_2 < attr_idx_3:
                    return f"{attr_name_1} + {attr_name_2} < {attr_name_3}"
                else:
                    return f"{attr_name_1} + {attr_name_2} ≥ {attr_name_3}"

        else:
            raise Exception("Multiplicative smaller than split is not initialized, hence "
                            "cannot explain a decision (Code: 23423423524)")

    def get_child_node_indices_for_sample(self, sample: np.ndarray) -> typing.List[int]:
        attr_idx_1, attr_idx_2, attr_idx_3 = self.get_split_attribute_indices()
        if sample[attr_idx_1] * sample[attr_idx_2] < attr_idx_3:
            return [0]
        else:
            return [1]

    def _get_children_indexer_and_state(self,
                                        data_values_left: np.ndarray,
                                        data_values_middle: np.ndarray,
                                        data_values_right: np.ndarray, *args, **kwargs) -> Optional[
                                            typing.List[typing.Tuple[typing.Dict,
                                                         typing.Tuple[np.ndarray]]]]:

        left_indices = (data_values_left + data_values_left) < data_values_right
        return [({}, (left_indices, ~left_indices))]


def get_available_split_rules(only_ui_rules: bool=True) -> typing.List[typing.Type[AbstractSplitRule]]:
    """
    Will return a list of classes of implemented split rules within THIS module

    :param only_ui_rules: Only return rules that the user should see within ui
    :return:
    """
    rules = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and not inspect.isabstract(obj) and issubclass(obj, AbstractSplitRule) \
                and not obj is FixedChainRule:

            if not only_ui_rules or obj.show_in_ui() is True:
                rules.append(obj)

    return rules


def get_class_by_name(name: str) -> Optional[typing.Type[AbstractSplitRule]]:
    """
    Will return the class object (not instance!) of a split rule given its name (see @get_name), if any
    :param name: 
    :return: 
    """
    rules = get_available_split_rules()
    filtered_rules = filter(lambda rule: rule.get_name() == name, rules)
    try:
        return next(filtered_rules)
    except:
        pass

    return None


def simplify_rules(rules: List[AbstractSplitRule],
                   sample: Optional[np.ndarray] = None,
                   model_to_sample: Optional[Dict[any, np.ndarray]]= None):
    """
    Will try to merge rules together to receive a list of more easy rules
    can either supply all samples with @param sample or have each model see another sample by using the mapping
    model_to_sample

    :param rules:
    :return:
    """
    # create a copy of the rules

    assert not (sample is not None and model_to_sample is not None), "If providing a sample it either always has to be" \
                                                                     "the same sample for each model or there has to" \
                                                                     "be a mapping from rule to model " \
                                                                     "(not both) (Code: 382742839)"
    curr_rules = [*rules]

    # try to match one by one
    for i in range(len(curr_rules)):
        if model_to_sample is not None:
            tree_i = rules[i].get_tree()
            sample_for_i = model_to_sample[tree_i]
        else:
            sample_for_i = sample
        for j in range(i+1, len(curr_rules)):
            if model_to_sample is not None:
                tree_j = rules[j].get_tree()
                sample_for_j = model_to_sample[tree_j]
            else:
                sample_for_j = sample

            rule_new = rules[i].__add__(rules[j], sample=sample_for_i, try_reverse=False, use_attribute_names=True)

            if rule_new == (rules[i], rules[j]):  # did not merge try to switch
                rule_new = rules[j].__add__(rules[i], sample=sample_for_j,
                                            try_reverse=False,
                                            use_attribute_names=True)

            # hit (we could merge)
            if isinstance(rule_new, AbstractSplitRule):
                # remove the consumed rules
                curr_rules.remove(rules[i])
                curr_rules.remove(rules[j])

                # add the new one
                curr_rules.append(rule_new)

                # and start all over
                return simplify_rules(curr_rules, model_to_sample=model_to_sample)

    return curr_rules