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
from hd_tree_classes.information_measure import RelativeAccuracyMeasure, EntropyMeasure
from hd_tree_classes.hdtree import HDTreeClassifier
from hd_tree_classes.split_rule import SmallerThanSplit, SingleCategorySplit, LessThanHalfOfSplit, \
    CloseToMedianSplit, FiveQuantileSplit, TenQuantileSplit, MedianSplit, TenQuantileMultiplicativeSplit, \
    MedianMultiplicativeQuantileSplit, FiveQuantileMultiplicativeSplit, FiveQuantileAdditiveSplit, TenQuantileAdditiveSplit, MedianAdditiveQuantileSplit
from sklearn.metrics import accuracy_score
import pandas as pd

if __name__ == '__main__':
    df_primitives = pd.read_csv(r"C:\Users\Richard\Dropbox\dev\hdtree\data\primitives.csv")
    y_primitives = df_primitives.loc[:, 'target'].values
    x_primitives = df_primitives.iloc[:, 1:-1].values

    hd_tree_primitives = HDTreeClassifier(allowed_splits=[
                                                        LessThanHalfOfSplit,
                                                        SingleCategorySplit,
                                                        SmallerThanSplit,
                                                        TenQuantileSplit,
                                                        CloseToMedianSplit,
                                                        MedianSplit,
                                                        TenQuantileMultiplicativeSplit,
                                                        FiveQuantileMultiplicativeSplit,
                                                        MedianMultiplicativeQuantileSplit,
                                                        TenQuantileAdditiveSplit,
                                                        TenQuantileAdditiveSplit,
                                                        FiveQuantileMultiplicativeSplit,
                                                        MedianAdditiveQuantileSplit
                                                    ],
                                   information_measure=EntropyMeasure(),
                                   max_levels=2, min_samples_at_leaf=None,
                                   verbose=True,
                                   attribute_names=[*df_primitives.columns][1:-1]
                                  )


    hd_tree_primitives.fit(x_primitives, y_primitives)

    y_pred = hd_tree_primitives.predict(x_primitives)
    print(accuracy_score(y_true=y_primitives, y_pred=hd_tree_primitives.predict(x_primitives)))
    print("Done")
    print(hd_tree_primitives)
    print(hd_tree_primitives.classes_,
          hd_tree_primitives.predict_proba(X=x_primitives[:2]),
          hd_tree_primitives.predict(X=x_primitives[:2]))

