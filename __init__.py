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
# @created: 29.10.2019
"""
from .src.hdtree import HDTreeClassifier
from .src.node import Node
from .src.information_measure import EntropyMeasure, RelativeAccuracyMeasure
from .src.split_rule import SmallerThanSplit, SingleCategorySplit, LessThanHalfOfSplit, \
    CloseToMedianSplit, FiveQuantileSplit, TenQuantileSplit, MedianSplit, TenQuantileMultiplicativeSplit, \
    MedianMultiplicativeQuantileSplit, FiveQuantileMultiplicativeSplit, FiveQuantileAdditiveSplit, \
    TenQuantileAdditiveSplit, MedianAdditiveQuantileSplit, TwentyQuantileAdditiveSplit, \
    TwentyQuantileMultiplicativeSplit, TwentyQuantileSplit, get_available_split_rules, AbstractSplitRule, \
    FixedValueSplit, get_class_by_name, FixedChainRule, TenQuantileRangeSplit, TwentyQuantileRangeSplit, simplify_rules

__all__ = ['HDTreeClassifier',
           'Node',
           'get_available_split_rules',
           'get_class_by_name',
           'simplify_rules',

           'EntropyMeasure',
           'RelativeAccuracyMeasure',
           'SmallerThanSplit',
           'SingleCategorySplit',
           'LessThanHalfOfSplit',
           'CloseToMedianSplit',
           'FiveQuantileSplit',
           'TwentyQuantileMultiplicativeSplit',
           'TwentyQuantileSplit',
           'TwentyQuantileAdditiveSplit',
           'TenQuantileSplit',
           'MedianSplit',
           'TenQuantileRangeSplit',
           'TwentyQuantileRangeSplit',
           'TenQuantileMultiplicativeSplit',
           'MedianMultiplicativeQuantileSplit',
           'FiveQuantileMultiplicativeSplit',
           'FiveQuantileAdditiveSplit',
           'TenQuantileAdditiveSplit',
           'MedianAdditiveQuantileSplit',
           'FixedValueSplit',

           'FixedChainRule',
           'AbstractSplitRule']