import os
from typing import Dict, List, Tuple

import pandas as pd
from pandas import DataFrame


datasets_dir = "datasets"

# functions in this module load datasets from the datasets dir; files should
# be in .csv format and function names *must* follow the convention:
# _load_DATASET_NAME() (as well as appropriate directories must also be named
# after datasets)


def _load_acute_inflammations() \
        -> Tuple[DataFrame, DataFrame, Dict]:
    file_path = os.path.join(datasets_dir, "acute_inflammations.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -2]
    X = dataset.iloc[:, :-2]

    metadata = dict()
    features_names = list(X.columns.values)
    categorical_features = [1, 2, 3, 4, 5]
    categorical_names = {
        1: {0: "False", 1: "True"},
        2: {0: "False", 1: "True"},
        3: {0: "False", 1: "True"},
        4: {0: "False", 1: "True"},
        5: {0: "False", 1: "True"}
    }
    class_names = ["healthy", "inflammation"]

    metadata["features_names"] = features_names
    metadata["categorical_features"] = categorical_features
    metadata["categorical_names"] = categorical_names
    metadata["class_names"] = class_names

    return X, y, metadata


def _load_breast_cancer_coimbra() \
        -> Tuple[DataFrame, DataFrame, Dict]:
    file_path = os.path.join(datasets_dir, "breast_cancer_coimbra.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]

    metadata = dict()
    features_names = list(X.columns.values)
    categorical_features = []
    categorical_names = {}
    class_names = ["healthy", "inflammation"]

    metadata["features_names"] = features_names
    metadata["categorical_features"] = categorical_features
    metadata["categorical_names"] = categorical_names
    metadata["class_names"] = class_names

    return X, y, metadata


def _load_breast_cancer_wisconsin() \
        -> Tuple[DataFrame, DataFrame, Dict]:
    file_path = os.path.join(datasets_dir, "breast_cancer_wisconsin.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]

    metadata = dict()
    features_names = list(X.columns.values)
    categorical_features = []
    categorical_names = {}
    class_names = ["healthy", "inflammation"]

    metadata["features_names"] = features_names
    metadata["categorical_features"] = categorical_features
    metadata["categorical_names"] = categorical_names
    metadata["class_names"] = class_names

    return X, y, metadata


def load_dataset(name: str) \
        -> Tuple[DataFrame, DataFrame, Dict]:
    function_name: str = "_load_" + name
    if function_name not in globals():
        raise ValueError(f"Function {function_name} not recognized")
    return globals()[function_name]()


