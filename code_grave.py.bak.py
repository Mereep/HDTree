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
# @created: 25.08.2019
"""

class NumericalSplit(AbstractNumericalSplit):
    """
    Will just split the guy at the numerical attribute tthat devides
    the samples into two halfes at the middle sample
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attribute: int = None
        self._used_separator_val: float = None

    def explain_split(self, sample: np.ndarray):
        if self._used_attribute is not None and self._used_separator_val:
            attr = sample[self._used_attribute]
            if attr is None:
                return f"\"{self.get_tree().get_attribute_names()[self._used_attribute]}\" is not available"
            else:
                if attr < self._used_separator_val:
                    return f"\"{self.get_tree().get_attribute_names()[self._used_attribute]}\" < " \
                           f"{round(self._used_separator_val,2)}"
                else:
                    return f"\"{self.get_tree().get_attribute_names()[self._used_attribute]}\" ≥ " \
                           f"{round(self._used_separator_val,2)}"
        else:
            raise Exception("Numerical split not initialized, hence, cannot explain a decision (Code: 23423423)")

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        if self._used_attribute is not None and self._used_separator_val:
            return f"{self.get_tree().get_attribute_names()[self._used_attribute]} < " \
                   f"{round(self._used_separator_val, 2)}"
        else:
            return "Numerical split not initialized"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr = sample[self._used_attribute]
        if attr is None:
            return self.get_child_nodes()
        else:
            if attr < self._used_separator_val:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        #node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        split_attribute = None
        split_value = None

        for attr_idx in supported_cols:
            # remove Nones since we cannot sort them
            numerical_col = node_data[:, attr_idx]
            numerical_col_not_none = self._filter_col_none(numerical_col)

            if not len(numerical_col_not_none) <= 1:
                # median of the list
                #sorted_members = np.sort(numerical_col_not_none)
                # split_member = sorted_members[len(sorted_members) // 2]
                split_member = np.median(numerical_col_not_none)

                #split_member = (split_member_left + split_member_right) / 2
                none_members_indexer = self._get_col_none_indexer(column=numerical_col)

                left_members_indices = node_indices[~none_members_indexer][numerical_col_not_none < split_member]
                right_members_indices = node_indices[~none_members_indexer][numerical_col_not_none >= split_member]

                if len(left_members_indices) == 0 or len(right_members_indices) == 0:
                    continue

                # put samples where the attribute is missing to both the left and the right side
                if np.any(none_members_indexer):
                    left_members_indices = np.append(left_members_indices, node_indices[none_members_indexer])
                    right_members_indices = np.append(right_members_indices, node_indices[none_members_indexer])

                self.set_child_nodes(child_nodes=[node.create_child_node(
                                                                 assigned_data_indices=left_members_indices),
                                                  node.create_child_node(
                                                                 assigned_data_indices=right_members_indices)])

                score = self.get_information_measure()(parent_node=self.get_node())

                if score > best_score:
                    best_score = score
                    best_children = self.get_child_nodes()
                    split_attribute = attr_idx
                    split_value = split_member

        if best_score > -float('inf'):
            self._used_attribute = split_attribute
            self._used_separator_val = split_value

            return best_score, best_children


        return None



class SingleCategorySplit(AbstractCategoricalSplit):
    """
    Will split on a single categorical attribute
    e.g. an attribute having 10 unique values will split in up tp 10 childs
    (Depending if node in question has all values inside)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attribute: int = None
        self._value_to_node_lookup: typing.Dict[str, 'Node'] = None

    def explain_split(self, sample: np.ndarray):
        if self._used_attribute is not None:
            attr = sample[self._used_attribute]

            attr_name = self.get_tree().get_attribute_names()[self._used_attribute]

            if attr is None:
                return f"\"{attr_name}\" is not available"
            elif attr not in self._value_to_node_lookup: # we did not split on that specific val
                print(self._value_to_node_lookup.keys())
                return f"\"{attr_name}\" with value {attr} was not available here during training"
            else:
                return f"\"{attr_name}\" is {attr}"

        else:
            raise Exception("Single Category Split Split not initialized, hence, "
                            "cannot explain a decision (Code: 23423423423423)")

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        attr_name = self.get_tree().get_attribute_names()[self._used_attribute]

        if self._used_attribute is not None and self._is_evaluated:
            return f"Split on categorical attribute \"{attr_name}\""
        else:
            return "Single Category Split is not evaluated"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr = sample[self._used_attribute]

        if attr is None or attr not in self._value_to_node_lookup:
            return self.get_child_nodes()
        else:
            return [self._value_to_node_lookup[attr]]

    def _get_best_split(self):
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()


        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        split_attribute = None
        best_value_to_node_lookup = None

        for attr_idx in supported_cols:
            categorical_col = node_data[:, attr_idx]
            none_indexer = self._get_col_none_indexer(column=categorical_col)
            valid_indexer = ~none_indexer

            # get all distinct not-None values
            available_node_labels = [*set(categorical_col[valid_indexer])]

            lookup_indices = {}
            for cat in available_node_labels:
                valid_data = node_data[valid_indexer, attr_idx]
                lookup_indices[cat] = node_indices[valid_indexer][valid_data == cat]

            child_nodes = []
            value_to_node_lookup = {}
            for cat, children in lookup_indices.items():
                if np.any(none_indexer):
                    children = np.append(children, node_indices[none_indexer])

                if len(children) >= 1:
                    child_node = node.create_child_node(assigned_data_indices=children)
                    value_to_node_lookup[cat] = child_node
                    child_nodes.append(child_node)


            # not too valid
            if len(child_nodes) <= 1:
                continue

            self.set_child_nodes(child_nodes=child_nodes)

            score = self.get_information_measure()(parent_node=self.get_node())

            # update score if best
            if score > best_score:
                best_score = score
                best_children = self.get_child_nodes()
                split_attribute = attr_idx
                best_value_to_node_lookup = value_to_node_lookup

        # if some valid split made, we update the guy
        if best_score > -float('inf'):
            self._used_attribute = split_attribute
            self._value_to_node_lookup = best_value_to_node_lookup
            return best_score, best_children

        return None


class CloseToMedianSplit(AbstractNumericalSplit):
    """
    Will split on a numerical attributes median +- 1/2 * stdev
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attribute: int = None
        self._median_val: float = None
        self._attribute_stddev: float = None

    def explain_split(self, sample: np.ndarray):
        if self._used_attribute is not None and self._median_val is not None:
            attr = sample[self._used_attribute]
            median = self._median_val
            stdev = self._attribute_stddev
            attr_name = self.get_tree().get_attribute_names()[self._used_attribute]

            if attr is None:
                return f"\"{self.get_tree().get_attribute_names()[self._used_attribute]}\" is not available"
            else:
                if abs(attr - median) <= 0.5 * stdev:
                    return f"\"{attr_name}\" is close to groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ² = {0.5 * round(stdev, 2)})"
                else:
                    return f"\"{attr_name}\" is outside of groups' median of  " \
                           f"{round(median, 2)} (± ½ × σ² = {0.5 * round(stdev, 2)})"
        else:
            raise Exception("Close To Median Split not initialized, hence, "
                            "cannot explain a decision (Code: 34534534534)")

    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        if self._used_attribute is not None and self._median_val is not None:
            return f"{self.get_tree().get_attribute_names()[self._used_attribute]} is close to the groups' median of " \
                   f"{round(self._median_val, 2)} (± ½ × σ² = {0.5 * round(self._attribute_stddev, 2)})"
        else:
            return "Numerical split not initialized 2"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr = sample[self._used_attribute]
        if attr is None:
            return self.get_child_nodes()
        else:
            if abs(self._median_val - attr) <= 0.5 * self._attribute_stddev:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        split_attribute = None
        split_value = None
        split_stdev = None

        for attr_idx in supported_cols:
            # remove Nones since we cannot sort them
            numerical_col = node_data[:, attr_idx]
            numerical_col_not_none = self._filter_col_none(numerical_col)

            if not len(numerical_col_not_none) <= 1:
                # median of the list
                # sorted_members = np.sort(numerical_col_not_none)
                # median_val = sorted_members[len(sorted_members) // 2]
                median_val = np.median(numerical_col_not_none)
                stdev = np.std(numerical_col_not_none)
                # stdev = np.std(sorted_members)

                inside_median = np.abs(numerical_col_not_none - median_val) <= 0.5 * stdev

                # split_member = (split_member_left + split_member_right) / 2
                none_members_indexer = self._get_col_none_indexer(column=numerical_col)

                left_members_indices = node_indices[~none_members_indexer][inside_median]
                right_members_indices = node_indices[~none_members_indexer][~inside_median]

                # check if we split useful at all
                if len(left_members_indices) == 0 or len(right_members_indices) == 0:
                    continue

                # put samples where the attribute is missing to both the left and the right side
                if np.any(none_members_indexer):
                    left_members_indices = np.append(left_members_indices, node_indices[none_members_indexer])
                    right_members_indices = np.append(right_members_indices, node_indices[none_members_indexer])

                self.set_child_nodes(child_nodes=[node.create_child_node(
                    assigned_data_indices=left_members_indices),
                    node.create_child_node(
                        assigned_data_indices=right_members_indices)])

                score = self.get_information_measure()(parent_node=self.get_node())

                if score > best_score:
                    best_score = score
                    best_children = self.get_child_nodes()
                    split_attribute = attr_idx
                    split_value = median_val
                    split_stdev = stdev

        if best_score > -float('inf'):
            self._used_attribute = split_attribute
            self._median_val = split_value
            self._attribute_stddev = split_stdev

            return best_score, best_children

        return None


class SmallerThanSplit(AbstractNumericalSplit):
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
        if self._used_attributes is not None and self._is_evaluated:
            return f"{self.get_tree().get_attribute_names()[self._used_attributes[0]]} < " \
                   f"{self.get_tree().get_attribute_names()[self._used_attributes[1]]}"
        else:
            return "Smaller than split not initialized"

    def explain_split(self, sample: np.ndarray):
        if self._used_attributes is not None:
            attr1 = sample[self._used_attributes[0]]
            attr2 = sample[self._used_attributes[1]]

            attr1_name = self.get_tree().get_attribute_names()[self._used_attributes[0]]
            attr2_name = self.get_tree().get_attribute_names()[self._used_attributes[1]]

            if attr1 is None:
                return f"\"{attr1_name}\" is not available"
            elif attr2 is None:
                return f"\"{attr1_name}\" is not available"
            else:
                if attr1 < attr2:
                    return f"\"{attr1_name}\" < \"{attr2_name}\""
                else:
                    return f"\"{attr1_name}\" ≥ \"{attr2_name}\""

        else:
            raise Exception("Smaller Than Split not initialized, hence, "
                            "cannot explain a decision (Code: 3453453434543)")

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr_1 = sample[self._used_attributes[0]]
        attr_2 = sample[self._used_attributes[1]]

        if attr_1 is None or attr_2 is None:
            return self.get_child_nodes()
        else:
            if attr_1 < attr_2:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        # node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        split_attributes = None

        for attr_idx_1 in supported_cols:
            for attr_idx_2 in supported_cols[attr_idx_1 + 1:]:

                numerical_col_1 = node_data[:, attr_idx_1]
                numerical_col_2 = node_data[:, attr_idx_2]

                none_indexer = (self._get_col_none_indexer(numerical_col_1) |
                                self._get_col_none_indexer(numerical_col_2))

                valid_indexer = ~ none_indexer

                if np.any(valid_indexer):
                    valid_cols_1 = numerical_col_1[valid_indexer]
                    valid_cols_2 = numerical_col_2[valid_indexer]

                    left_members_indices = node_indices[valid_indexer][valid_cols_1 < valid_cols_2]
                    right_members_indices = node_indices[valid_indexer][valid_cols_1 >= valid_cols_2]

                    # if we do not gain information leave here
                    if len(left_members_indices) == 0 or len(right_members_indices) == 0:
                        continue

                    # put samples where the attribute is missing to both the left and the right side
                    if np.any(none_indexer):
                        # none_members_indices = node_indices[~valid_indexer]
                        left_members_indices = np.append(left_members_indices, node_indices[none_indexer])
                        right_members_indices = np.append(right_members_indices, node_indices[none_indexer])

                    self.set_child_nodes(child_nodes=[node.create_child_node(
                        assigned_data_indices=left_members_indices),
                        node.create_child_node(
                            assigned_data_indices=right_members_indices)])

                    score = self.get_information_measure()(parent_node=self.get_node())

                    if score > best_score:
                        best_score = score
                        best_children = self.get_child_nodes()
                        split_attributes = (attr_idx_1, attr_idx_2)

        if best_score > -float('inf'):
            self._used_attributes = split_attributes
            return best_score, best_children

        return None

class LessThanHalfOfSplit(AbstractNumericalSplit):
    """
    Splits on a1 < 1/2 * a2 rule
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_attributes: typing.Tuple[int, int] = None

    def explain_split(self, sample: np.ndarray):
        if self._used_attributes is not None:
            attr1 = sample[self._used_attributes[0]]
            attr2 = sample[self._used_attributes[1]]

            attr1_name = self.get_tree().get_attribute_names()[self._used_attributes[0]]
            attr2_name = self.get_tree().get_attribute_names()[self._used_attributes[1]]

            if attr1 is None:
                return f"\"{attr1_name}\" is not available"
            elif attr2 is None:
                return f"\"{attr1_name}\" is not available"
            else:
                if attr1 < 0.5 * attr2:
                    return f"\"{attr1_name}\" < ½ × {attr2_name}"
                else:
                    return f"\"{attr1_name}\" ≥ ½ × {attr2_name}"

        else:
            raise Exception("Less Than Half Of Split not initialized, hence "
                            "cannot explain a decision (Code: 23423234256)")


    def user_readable_description(self):
        """
        Will return what the split was about
        :return:
        """
        if self._used_attributes is not None and self._is_evaluated:
            return f"{self.get_tree().get_attribute_names()[self._used_attributes[0]]} is less than half of " \
                   f"{self.get_tree().get_attribute_names()[self._used_attributes[1]]}"
        else:
            return "Smaller than split not initialized"

    def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
        super().get_child_nodes_for_sample(sample=sample)
        attr_1 = sample[self._used_attributes[0]]
        attr_2 = sample[self._used_attributes[1]]

        if attr_1 is None or attr_2 is None:
            return self.get_child_nodes()
        else:
            if attr_1 < 0.5 * attr_2:
                return [self.get_child_nodes()[0]]
            else:
                return [self.get_child_nodes()[1]]

    def _get_best_split(self):
        node = self.get_node()
        node_data = node.get_data()
        node_indices = node.get_data_indices()
        supported_cols = self._get_attribute_candidates()

        if len(supported_cols) == 0:
            return None

        #node_dummy = node.__class__(tree=node.get_tree(), assigned_data_indices=node.get_data_indices())
        best_score = -float("inf")
        best_children = []
        split_attributes = None


        for attr_idx_1 in supported_cols:
            for attr_idx_2 in supported_cols[attr_idx_1+1:]:

                numerical_col_1 = node_data[:, attr_idx_1]
                numerical_col_2 = node_data[:, attr_idx_2]

                none_indexer = (self._get_col_none_indexer(numerical_col_1) |
                                   self._get_col_none_indexer(numerical_col_2))

                valid_indexer = ~ none_indexer

                if np.any(valid_indexer):
                    valid_cols_1 = numerical_col_1[valid_indexer]
                    valid_cols_2 = numerical_col_2[valid_indexer]

                    left_members_indices = node_indices[valid_indexer][valid_cols_1 < 0.5 * valid_cols_2]
                    right_members_indices = node_indices[valid_indexer][valid_cols_1 >= 0.5 * valid_cols_2]

                    # if we do not gain information leave here
                    if len(left_members_indices) == 0 or len(right_members_indices) == 0:
                        continue

                    # put samples where the attribute is missing to both the left and the right side
                    if np.any(none_indexer):
                        #none_members_indices = node_indices[~valid_indexer]
                        left_members_indices = np.append(left_members_indices, node_indices[none_indexer])
                        right_members_indices = np.append(right_members_indices, node_indices[none_indexer])

                    self.set_child_nodes(child_nodes=[node.create_child_node(assigned_data_indices=left_members_indices),
                                                      node.create_child_node(assigned_data_indices=right_members_indices)])

                    score = self.get_information_measure()(parent_node=self.get_node())

                    if score > best_score:
                        best_score = score
                        best_children = self.get_child_nodes()
                        split_attributes = (attr_idx_1, attr_idx_2)

        if best_score > -float('inf'):
            self._used_attributes = split_attributes
            return best_score, best_children

        return None



# class NumericalSplit(AbstractNumericalSplit, OneAttributeSplitMixin):
#
#     def explain_split(self, sample: np.ndarray):
#         state = self.get_state()
#         if state is not None:
#             attr_name = self.get_split_attribute_name()
#             split_val = state['split_value']
#             attr_index = self.get_split_attribute_index()
#
#             attr = sample[attr_index]
#
#             if attr is None:
#                 return f"Attribute \"{attr_name}\" is not available"
#             else:
#                 if attr < split_val:
#                     return f"\"{attr_name}\" < " \
#                            f"{round(split_val, 2)}"
#                 else:
#                     return f"\"{attr_name}\" ≥ " \
#                            f"{round(split_val, 2)}"
#         else:
#             raise Exception("Numerical split not initialized, hence, cannot explain a decision (Code: 8903485768930)")
#
#     def user_readable_description(self):
#         """
#         Will return what the split was about
#         :return:
#         """
#         state = self.get_state()
#
#         if state is not None:
#             attr_name = self.get_split_attribute_name()
#             split_val = state['split_value']
#             return f"{attr_name} < " \
#                    f"{round(split_val, 2)}"
#         else:
#             return "Numerical split not initialized"
#
#     def get_child_nodes_for_sample(self, sample: np.ndarray) -> typing.List['Node']:
#         super().get_child_nodes_for_sample(sample=sample)
#         state = self.get_state()
#         val = state['split_value']
#         attr_idx = self.get_split_attribute_index()
#
#         attr = sample[attr_idx]
#         if attr is None:
#             return self.get_child_nodes()
#         else:
#             if attr < val:
#                 return [self.get_child_nodes()[0]]
#             else:
#                 return [self.get_child_nodes()[1]]
#
#     def _get_children_indexer_and_state(self, data_values: np.ndarray):
#         labels = self.get_node().get_labels()
#         vals_labels = np.c_[data_values, labels]
#         sorted_vals_labels = np.sort(vals_labels, axis=0)
#         best = float('inf')
#         split_val = None
#         measure = self.get_tree().get_information_measure()
#         sample_cnt = len(data_values)
#
#         for i in range(len(sorted_vals_labels)):
#
#             value_left = measure.calculate_for_labels(sorted_vals_labels[:1, 1], normalize=False)
#             value_right = measure.calculate_for_labels(sorted_vals_labels[1:, 1], normalize=False)
#
#             val = (i/sample_cnt) * value_left + ((sample_cnt-i)/sample_cnt) * value_right
#
#             if val < best:
#                 best = val
#                 split_val = sorted_vals_labels[i, 0]
#
#         if split_val is not None:
#             left = data_values < split_val
#             return [({'split_value': split_val}, (left, ~left))]
#
#         return None
#
#         # vals_with_idx = np.c_[data_values, np.arange(0, len(data_values), 1)]
#         #for val_idx in enumerate(vals_with_idx):
#         #   idx = val_idx[0]
#             #left = data_values < val_idx[0]
#             #right = ~left
#             #state = {'split_value': split_member}
#
#         #    results.append((state, (left, right)))
#
#         #return results

