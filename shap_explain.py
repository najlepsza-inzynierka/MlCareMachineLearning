from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def explain_prediction_with_image(
        clf: xgb.XGBClassifier,
        x: pd.DataFrame) \
        -> np.array:
    """
    Creates SHAP force plot to explain XGBoost prediction for a particular
    sample x.

    :param clf: trained XGBoost classifier
    :param x: Pandas DataFrame with sample that we want to explain
    :return: 3D Numpy array with RGB image of SHAP force plot
    """
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x.to_numpy())

    fig = shap.force_plot(explainer.expected_value,
                          shap_values=shap_values,
                          features=x,
                          show=False,
                          matplotlib=True)

    # "save" file to bytes buffer, since .savefig() version looks best and
    # this way we don't do disk write/read
    buffer = BytesIO()
    fig.savefig(buffer,
                format="png",
                dpi=150,
                bbox_inches='tight')
    buffer.seek(0)
    buffer = buffer.read()

    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
