from operator import itemgetter
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def get_feature_importances(X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series, pd.DataFrame]) \
                            -> List[Tuple[str, float]]:
    """
    Calculates feature importances for a given dataset and returns them,
    sorted descending. If names of features are not available, dummy values
    "feature0", "feature1" etc. will be used instead.

    :param X: array-like, data matrix; if it's a Pandas dataframe, the feature
    names from it will be used
    :param y: vector-like, class vector
    :return: list of tuples (feature name, feature importance), sorted
    descending by feature importances
    """
    clf = ExtraTreesClassifier(n_estimators=500,
                               max_features=None,
                               n_jobs=-1,
                               random_state=0)
    clf.fit(X, y)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.values
    else:
        feature_names = ["feature" + str(i) for i in range(X.shape[1])]
    feature_importances = clf.feature_importances_
    result = [(str(name), importance)
              for name, importance
              in zip(feature_names, feature_importances)]
    result.sort(key=itemgetter(1), reverse=True)
    return result



