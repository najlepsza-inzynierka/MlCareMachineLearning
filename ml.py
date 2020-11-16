import math
from operator import itemgetter
from typing import Dict, List, Tuple, Union

import cv2
import graphviz
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, \
    StratifiedKFold
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import xgboost as xgb


def get_feature_importances(X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series, pd.DataFrame]) \
        -> List[Tuple[str, float]]:
    """
    Calculates feature importances for a given dataset and returns them,
    sorted descending. If names of features are not available, dummy values
    "feature0", "feature1" etc. will be used instead.

    :param X: array-like, data matrix; if it"s a Pandas dataframe, the feature
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


def train_decision_tree(X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series, pd.DataFrame]) \
        -> Tuple[DecisionTreeClassifier, Dict[str, float]]:
    """
    Train the decision tree classifier on the given dataset, automatically
    choosing the optimal hyperparameters through grid search cross-validation.

    :param X: matrix-like, samples matrix
    :param y: vector-like, classes of samples
    :return: trained classifier and dictionary with metric scores from test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier(random_state=0)

    # deeper trees would not be easily interpretable in graphical form
    depths = range(1, 8)
    min_samples = range(2, int(math.sqrt(X.shape[0])))
    class_weights = [None, "balanced"]
    ccp_alphas = clf.cost_complexity_pruning_path(X_train, y_train).ccp_alphas

    param_grid = {"max_depth": depths,
                  "min_samples_split": min_samples,
                  "min_samples_leaf": min_samples,
                  "class_weight": class_weights,
                  "ccp_alpha": ccp_alphas}

    tree = GridSearchCV(DecisionTreeClassifier(random_state=0),
                        param_grid,
                        n_jobs=-1,
                        verbose=1)
    tree.fit(X_train, y_train)

    tree = tree.best_estimator_
    y_pred = tree.predict(X_test)
    scores = {"accuracy": round(accuracy_score(y_test, y_pred), 2),
              "precision": round(precision_score(y_test, y_pred), 2),
              "recall": round(recall_score(y_test, y_pred), 2),
              "f1": round(f1_score(y_test, y_pred), 2),
              "roc_auc": round(roc_auc_score(y_test, y_pred), 2)}

    return tree, scores


def get_decision_tree_plot(
        clf: DecisionTreeClassifier,
        feature_names: Union[List[str], None],
        class_names: Union[List[str], None]) \
        -> np.ndarray:
    """
    Plots the decision tree and returns the image as a Numpy array.

    :param clf: decision tree classifier to be plotted
    :param feature_names: list of names of features to be plotted or None
    :param class_names: list of names of classes to be plotted or None
    :return: Numpy array with the plot
    """
    # if/else make sure those are Nones, not empty lists
    feature_names = feature_names if feature_names else None
    class_names = class_names if class_names else None

    tree: str = export_graphviz(decision_tree=clf,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                label="all",
                                filled=True,
                                impurity=False,
                                proportion=True,
                                rounded=True,
                                precision=2)

    # get tree plot in memory as bytes of image
    image_bytes: bytes = graphviz.Source(tree).pipe(format="png")

    # decode bytes as image; OpenCV uses BGR, convert to RGB for right display
    image: np.ndarray = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def train_XGBoost(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]) \
        -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Train the XGBoost classifier on the given dataset, automatically
    choosing the optimal hyperparameters through grid search cross-validation.

    :param X: matrix-like, samples matrix
    :param y: vector-like, classes of samples
    :return: trained classifier and dictionary with metric scores from test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    base_estimator = xgb.XGBClassifier(eval_metric='auc', n_jobs=-1)
    scoring = "roc_auc"

    search_spaces = {
            "learning_rate": Real(0.01, 1.0, "log-uniform"),  # eta
            "min_split_loss": Real(1e-3, 0.5, "log-uniform"),  # gamma
            "max_depth": Integer(1, 50),
            "min_child_weight": Real(0.5, 10, "log-uniform"),
            "subsample": Real(0.5, 1.0, "uniform"),
            "colsample_bytree": Real(0.5, 1.0, "uniform"),
            "colsample_bynode": Real(0.5, 1.0, "uniform"),
            "lambda": Real(1e-3, 1000, "log-uniform"),  # L2 regularization
            "alpha": Real(1e-3, 1.0, "log-uniform"),  # L1 regularization
            "n_estimators": Integer(50, 100)
    }

    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    bayesian_optimizer = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=search_spaces,
        scoring=scoring,
        cv=CV,
        n_jobs=-1,
        n_iter=50,
        random_state=0
    )

    bayesian_optimizer.fit(X_train, y_train)
    xgboost = bayesian_optimizer.best_estimator_

    y_pred = xgboost.predict(X_test)
    scores = {"accuracy": round(accuracy_score(y_test, y_pred), 2),
              "precision": round(precision_score(y_test, y_pred), 2),
              "recall": round(recall_score(y_test, y_pred), 2),
              "f1": round(f1_score(y_test, y_pred), 2),
              "roc_auc": round(roc_auc_score(y_test, y_pred), 2)}

    return xgboost, scores
