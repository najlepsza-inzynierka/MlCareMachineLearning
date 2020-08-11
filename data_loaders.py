import os
import pandas as pd
from pandas import DataFrame
from typing import Tuple


datasets_dir = "datasets"


def load_acute_inflammations() -> Tuple[DataFrame, DataFrame]:
    file_path = os.path.join(datasets_dir, "acute_inflammations.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -2]
    X = dataset.iloc[:, :-2]
    return X, y


def load_breast_cancer_coimbra() -> Tuple[DataFrame, DataFrame]:
    file_path = os.path.join(datasets_dir, "breast_cancer_coimbra.csv")
    dataset = pd.read_csv(file_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_breast_cancer_wisconsin() -> Tuple[DataFrame, DataFrame]:
    file_path = os.path.join(datasets_dir, "breast_cancer_wisconsin.csv")
    dataset = pd.read_csv(file_path)

    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_dataset(name: str) -> Tuple[DataFrame, DataFrame]:
    function_name: str = "_load_" + name
    if function_name not in locals():
        raise ValueError(f"Function {function_name} not recognized")
    return locals()[function_name]()


