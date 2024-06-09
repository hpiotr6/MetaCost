import pickle
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from itertools import product
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.metacost import MetaCost


def eval(
    results,
    dataset_name,
    classifier_name,
    X_test,
    y_test,
    cost_matrix,
    model,
    model_type,
    elapsed_time,
):
    id = "-".join((dataset_name, classifier_name, model_type))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    results[id]["sum"] = np.sum(cost_matrix * cm)
    results[id]["f1"] = f1_score(y_test, y_pred, average="weighted")
    results[id]["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
    results[id]["precision"] = precision_score(y_test, y_pred, average="weighted")
    results[id]["recall"] = recall_score(y_test, y_pred, average="weighted")
    results[id]["time"] = elapsed_time


def run_exps(
    cost_function,
    data_factory,
    classifiers,
    m=10,
    n_ratio=10,
    p=False,
    q=False,
    save_name=None,
):
    results = defaultdict(dict)
    for dataset_name, (classifier_name, classifier) in tqdm(
        product(
            data_factory.datasets,
            classifiers.items(),
        ),
        total=len(data_factory.datasets) * len(classifiers),
        desc="Processing..",
    ):
        tqdm.write(f"Dataset: {dataset_name}, Classifier: {classifier_name}")

        X_train, y_train, X_test, y_test = data_factory.create_dataset(dataset_name)
        if "scipy.sparse._csr.csr_matrix" in str(type(X_test)):
            X_train, X_test = X_train.toarray(), X_test.toarray()

        num_classes = len(np.unique(y_test))
        num_examples = X_train.shape[0]

        cost_matrix = cost_function(y_train)
        partial_fit_predict = partial(
            eval,
            results,
            dataset_name,
            classifier_name,
            X_test,
            y_test,
            cost_matrix,
        )

        metacost_model = clone(classifier)

        start_time = time.time()

        model = MetaCost(
            model=metacost_model,
            m=m,
            n=num_examples // n_ratio,
            num_classes=num_classes,
            p=p,
            q=q,
            C=cost_matrix,
        ).fit(X_train, y_train)

        end_time = time.time()

        elapsed_time = end_time - start_time

        partial_fit_predict(model, "metacost", elapsed_time)

        base_model = clone(classifier)

        start_time = time.time()
        base_model.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time

        partial_fit_predict(base_model, "baseline", elapsed_time)

    if not save_name:
        save_name = cost_function.__name__

    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)

    return results
