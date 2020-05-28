import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


dataset_dir = "./datasets/"
model_dir = "./models/"

dataset = pd.read_csv(dataset_dir + "acute_inflammations.csv", sep=r"\s+", header=None)
dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1, inplace=True)
dataset = dataset.infer_objects()

labels = dataset.iloc[:, -1]
dataset.drop(dataset.columns[len(dataset.columns)-1], axis=1, inplace=True)

dataset[0] = [x.replace(',', '.') for x in dataset[0]]
dataset[0] = dataset[0].astype(float)

booleanDictionary = {"yes": True, "no": False}
for column in dataset:
    if column == 0:
        continue
    dataset[column] = dataset[column].map(booleanDictionary)

print(dataset.head())

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.15, random_state=0)


best_param = 0
best_train = 0
best_test = 0
best_model = None
for param in [0.01]:
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    #model.save_model("acute_inflammations.model")

    train = accuracy_score(y_train, model.predict(X_train))
    test = accuracy_score(y_test, model.predict(X_test))

    if test > best_test:
        best_param = param
        best_train = train
        best_test = test
        best_model = model

print(best_param, best_train, best_test)

