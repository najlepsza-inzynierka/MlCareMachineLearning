import json
import os

import dill

from ml import *
from data_loaders import load_dataset

datasets_names = ["acute_inflammations", "breast_cancer_coimbra",
                  "breast_cancer_wisconsin"]


def calculate_feature_importances():
    for dataset in datasets_names:
        X, y, class_names = load_dataset(dataset)
        feature_importances = get_feature_importances(X, y)
        result = dict()
        result["all"] = feature_importances
        if len(feature_importances) >= 6:
            # enough features to populate both top_3 and last_3 with empty
            # intersection
            result["top_3"] = feature_importances[:3]
            result["last_3"] = feature_importances[-3:]
        elif len(feature_importances) in {4, 5}:
            # there are 4 or 5 features - we have enough to populate the top_3,
            # but have to limit the last_3
            result["top_3"] = feature_importances[:3]
            result["last_3"] = feature_importances[3:]
        else:
            # there are not even enough features to fill the top_3
            result["top_3"] = feature_importances
            result["last_3"] = []
        file_location = os.path.join(dataset, "feature_importances.json")
        with open(file_location, "w") as file:
            json.dump(result, file, indent=2)


def make_decision_trees():
    for dataset in datasets_names:
        X, y, metadata = load_dataset(dataset)
        clf, scores = train_decision_tree(X, y)

        with open(os.path.join(dataset, "tree_metrics.json"), "w") \
                as metrics_file:
            dill.dump(clf, os.path.join(dataset, "tree_model.joblib"))
            json.dump(scores, metrics_file, indent=2)

        plot = get_decision_tree_plot(
            clf,
            feature_names=metadata["feature_names"],
            class_names=metadata["class_names"])
        file_location = os.path.join(dataset, "tree.png")
        cv2.imwrite(file_location, plot)


def make_XGBoost_classifiers():
    for dataset in datasets_names:
        X, y, _ = load_dataset(dataset)
        clf, scores = train_XGBoost(X, y)

        clf.save_model(os.path.join(dataset, "xgboost_model.xgb"))
        with open(os.path.join(dataset, "xgboost_metrics.json"), "w") \
                as metrics_file:
            json.dump(scores, metrics_file, indent=2)


if __name__ == "__main__":
    #calculate_feature_importances()
    #make_decision_trees()
    make_XGBoost_classifiers()
