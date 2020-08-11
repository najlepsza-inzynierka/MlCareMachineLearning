from operator import itemgetter
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def get_feature_importances(X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series, pd.DataFrame]) \
                            -> List[Tuple[str, float]]:
    clf = ExtraTreesClassifier(n_estimators=500,
                               max_features=None,
                               n_jobs=-1,
                               random_state=0)
    clf.fit(X, y)
    feature_names = X.columns.values
    feature_importances = clf.feature_importances_
    result = [(str(name), importance)
              for name, importance
              in zip(feature_names, feature_importances)]
    result.sort(key=itemgetter(1), reverse=True)
    return result



