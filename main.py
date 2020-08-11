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
        result["top_3"] = feature_importances[:3]
        result["last_3"] = feature_importances[-3:]
        file_location = os.path.join(dataset, "feature_imporstances.json")
        with open(file_location, "w") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    calculate_feature_importances()
