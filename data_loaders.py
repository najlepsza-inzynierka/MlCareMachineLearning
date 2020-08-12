import os
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame


datasets_dir = "datasets"

# functions in this module load datasets from the datasets dir; files should
# be in .csv format and function names *must* follow the convention:
# _load_DATASET_NAME() (as well as appropriate directories must also be named
# after datasets)


def _load_acute_inflammations() -> Tuple[DataFrame, DataFrame, List[str]]:
    file_path = os.path.join(datasets_dir, "acute_inflammations.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -2]
    X = dataset.iloc[:, :-2]
    class_names = ["healthy", "inflammation"]
    return X, y, class_names


def _load_breast_cancer_coimbra() -> Tuple[DataFrame, DataFrame, List[str]]:
    file_path = os.path.join(datasets_dir, "breast_cancer_coimbra.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    class_names = ["healthy", "cancer"]
    return X, y, class_names


def _load_breast_cancer_wisconsin() -> Tuple[DataFrame, DataFrame, List[str]]:
    file_path = os.path.join(datasets_dir, "breast_cancer_wisconsin.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    class_names = ["healthy", "cancer"]
    return X, y, class_names


def load_dataset(name: str) -> Tuple[DataFrame, DataFrame, List[str]]:
    function_name: str = "_load_" + name
    if function_name not in locals():
        raise ValueError(f"Function {function_name} not recognized")
    return locals()[function_name]()


