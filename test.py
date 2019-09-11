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
import numpy as np
from hd_tree_classes.information_measure import RelativeAccuracyMeasure, EntropyMeasure
from sklearn.tree import DecisionTreeClassifier
import random
from hd_tree_classes.hdtree import HDTreeClassifier
from hd_tree_classes.split_rule import SmallerThanSplit, SingleCategorySplit, LessThanHalfOfSplit, \
    CloseToMedianSplit, FiveQuantileSplit, TenQuantileSplit, MedianSplit
from sklearn.metrics import accuracy_score
import pandas as pd

if __name__ == '__main__':
    np.random.seed(1)
    x_data_decision_tree = np.array([[np.random.rand(1)[0] * 10 - 5 + np.random.normal(scale=2),
                                      np.random.rand(1)[0] * 10 - 5 + np.random.normal(scale=2)] for x in range(1000)])


    # shift data a bit away from the street
    def shift(x, y):
        street_size = 0.5
        if abs(x - y) < 2*street_size:  # almost on boundary
            if x < y:
                x -= street_size
                y += street_size
            else:
                y -= street_size
                x += street_size
        return [x, y]


    x_data_decision_tree = np.array([*map(lambda xy: shift(*xy), x_data_decision_tree)])

    y_data_decision_tree = np.array([*map(lambda pos: 1 if pos[0] < pos[1] else 0, x_data_decision_tree)])

    np.random.seed(101)
    X_data = np.random.rand(10000, 2).astype(np.object)
    X_data = np.append(X_data, np.full(shape=(len(X_data), 1), fill_value='a'), axis=-1)
    y_data = np.array([str(random.randint(0,1)) for i in range(len(X_data))])

    tree = HDTreeClassifier(allowed_splits=[LessThanHalfOfSplit],
                           information_measure=RelativeAccuracyMeasure(),
                           max_levels=3,
                           min_samples_at_leaf=2,
                           verbose=True)


    tree.fit(x_data_decision_tree, y_data_decision_tree)
    print(accuracy_score(y_true=y_data_decision_tree, y_pred=tree.predict(x_data_decision_tree)))
    print(tree)


    df_titanic = pd.read_csv("C:/Users/Richard/Dropbox/dev/gang_of_nerds/data/titanic/train.csv")
    df_titanic_ok = df_titanic.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
    y_titanic = df_titanic.loc[:, 'Survived'].values
    x_titanic = df_titanic_ok.values

    hd_tree_titanic = HDTreeClassifier(allowed_splits=[
                                                       LessThanHalfOfSplit,
                                                       SingleCategorySplit,
                                                       SmallerThanSplit,
                                                       TenQuantileSplit,
                                                       CloseToMedianSplit,
                                                       MedianSplit,
                                                     ],
                                       information_measure=EntropyMeasure(),
                                       max_levels=None, min_samples_at_leaf=2,
                                       verbose=True, attribute_names=[*df_titanic_ok.columns])


    hd_tree_titanic.fit(x_titanic, y_titanic)

    print(hd_tree_titanic.explain_decision(x_titanic[1]))
    print(hd_tree_titanic)
    print(accuracy_score(y_true=y_titanic, y_pred=hd_tree_titanic.predict(x_titanic)))


    #
    #
    # x_random = np.random.rand(20000, 30)
    # y_random = ((np.random.rand(20000)*10) // 2 == 0)
    #
    #
    # hd_tree_random = HDTreeClassifier(allowed_splits=[
    #                                                    LessThanHalfOfSplit,
    #                                                    SingleCategorySplit,
    #                                                    SmallerThanSplit,
    #                                                    #NumericalSplit,
    #                                                    TenQuantileSplit,
    #                                                    CloseToMedianSplit
    #                                                  ],
    #                                    information_measure=EntropyMeasure(),
    #                                    max_levels=6, min_samples_at_leaf=3,
    #                                    verbose=True)
    #
    # hd_tree_random.fit(x_random, y_random)
    # print(hd_tree_random.explain_decision(x_random[1]))
    #
    # print(hd_tree_random)
    # print(accuracy_score(y_true=y_random, y_pred=hd_tree_random.predict(x_random)))


