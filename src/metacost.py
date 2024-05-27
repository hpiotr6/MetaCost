import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone


class MetaCost:
    def __init__(self, model, m, n, num_classes, p, q, C):
        self.model = model
        self.m_resamples = m
        self.n_examples = n
        self.num_classes = num_classes
        self.has_class_proba = p
        self.use_all_resamples = q
        self.cost_matrix = C

    def fit(self, X: np.array, y: np.array):
        y_new = self._y_relabel(X, y)
        assert y.shape == y_new.shape

        return self.model.fit(X, y_new)

    def _y_relabel(self, X: np.array, y: np.array) -> np.array:
        resampled_subsets = []
        _models = []
        for _ in range(self.m_resamples):
            chosen_ids = np.random.choice(X.shape[0], self.n_examples)
            _model = clone(self.model)
            _model.fit(X[chosen_ids], y[chosen_ids])

            resampled_subsets.append(chosen_ids)
            _models.append(_model)

        labels = []
        for i, x in enumerate(X):
            if not self.use_all_resamples:
                ids_to_leave = [i for subset in resampled_subsets if i not in subset]
                models = [
                    model
                    for idx, model in enumerate(_models)
                    if idx not in ids_to_leave
                ]
            else:
                models = _models

            if self.has_class_proba:
                p_jx_ms = np.asarray(
                    [model.predict_proba(x.reshape(1, -1)) for model in models]
                )
            else:
                preds = np.array([model.predict(x) for model in models])
                p_jx_ms = np.zeros((len(models), self.num_classes))
                p_jx_ms[np.arange(len(models)), preds] = 1

            p_jx = p_jx_ms.mean(0)
            label = np.argmin(np.dot(p_jx, self.cost_matrix))
            labels.append(label)

        return np.asarray(labels)
