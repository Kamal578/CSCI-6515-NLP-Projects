from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer


def get_models() -> dict[str, object]:
    return {
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "bernoulli_nb": BernoulliNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            solver="saga",
            random_state=42,
        ),
    }


def fit_predict(model_name: str, model, X_train, y_train, X_test):
    # BernoulliNB expects binary indicators, while the other models use raw count/aggregate values.
    if model_name == "bernoulli_nb":
        binarizer = Binarizer(copy=True)
        X_train_fit = binarizer.fit_transform(X_train)
        X_test_fit = binarizer.transform(X_test)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    model.fit(X_train_fit, y_train)
    return model.predict(X_test_fit)
