import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier

# load data
data_dir = "./data"
data_path = os.path.join(data_dir, 'cleveland_processed.csv')

df = pd.read_csv(data_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def calculate_accuracy_from_cm(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    train_acc = calculate_accuracy_from_cm(train_cm)
    test_acc = calculate_accuracy_from_cm(test_cm)

    print(f"{model_name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return model


def train_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                               algorithm='auto', leaf_size=30, p=2, metric='minkowski')
    return train_and_evaluate(knn, X_train, X_test, y_train, y_test, "KNN")


def train_svm(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', random_state=42)
    return train_and_evaluate(svm, X_train, X_test, y_train, y_test, "SVM")


def train_naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    return train_and_evaluate(nb, X_train, X_test, y_train, y_test, "Naive Bayes")


def train_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(
        criterion='gini', max_depth=10, min_samples_split=2)
    return train_and_evaluate(dt, X_train, X_test, y_train, y_test, "Decision Tree")


def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        criterion='gini', max_depth=10, min_samples_split=2, n_estimators=10, random_state=42)
    return train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest")


def train_adaboost(X_train, X_test, y_train, y_test):
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
    return train_and_evaluate(ada, X_train, X_test, y_train, y_test, "AdaBoost")


def train_gradient_boost(X_train, X_test, y_train, y_test):
    gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,
                                    subsample=1.0, min_samples_split=2, max_depth=3, random_state=42)
    return train_and_evaluate(gb, X_train, X_test, y_train, y_test, "Gradient Boost")


def train_xgboost(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(objective="binary:logistic",
                        random_state=42, n_estimators=100)
    return train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost")


def train_stacking(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier(random_state=42)
    gc = GradientBoostingClassifier(random_state=42)
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    ad = AdaBoostClassifier(random_state=42)

    stacking = StackingClassifier(
        classifiers=[dtc, rfc, knn, xgb, gc, svc, ad],
        meta_classifier=RandomForestClassifier(random_state=42),
        verbose=1,
    )

    return train_and_evaluate(stacking, X_train, X_test, y_train, y_test, "Stacking")

# Main function to train all models


def train_all_models(X_train, X_test, y_train, y_test):
    models = [
        train_knn,
        train_svm,
        train_naive_bayes,
        train_decision_tree,
        train_random_forest,
        train_adaboost,
        train_gradient_boost,
        train_xgboost,
        train_stacking
    ]

    trained_models = []
    for model_func in models:
        trained_models.append(model_func(X_train, X_test, y_train, y_test))

    return trained_models


# Usage example:
# Assuming you have your data split into X_train, X_test, y_train, y_test
trained_models = train_all_models(X_train, X_test, y_train, y_test)
