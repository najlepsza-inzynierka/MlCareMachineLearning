import json
import os

from commons import get_feature_importances
import data_loaders

datasets_names = ["acute_inflammations", "breast_cancer_coimbra",
                  "breast_cancer_wisconsin"]


def calculate_feature_importances():
    for dataset in datasets_names:
        loader_fun = getattr(data_loaders, "load_" + dataset)
        X, y = loader_fun()
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


if __name__ == "__main__":
    calculate_feature_importances()
