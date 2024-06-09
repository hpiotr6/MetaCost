from functools import cache
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class DataFactory:
    def __init__(
        self, datasets, train_test_split=0.2, random_state=42, fast_version_split=None
    ):
        self.train_test_split = train_test_split

        self._datasets = datasets
        self.datasets = list(self._datasets.keys())

        self.random_state = random_state
        self.fast_version_split = fast_version_split

    @cache
    def create_dataset(self, name):
        dataset = self._datasets[name]

        X = dataset.data.features
        y = dataset.data.targets

        if name == "adults":
            y.loc[:, "income"] = y["income"].str.replace(".", "")

        y_train, y_test, X_train, X_test = self.prepare_data(X, y)

        if self.fast_version_split and name != "wine_quality":
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, test_size=1.0 - self.fast_version_split
            )

        return X_train, y_train, X_test, y_test

    def prepare_data(self, X, y):
        lbl_enc = LabelEncoder()
        y = lbl_enc.fit_transform(y.values.ravel())
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=self.train_test_split, random_state=self.random_state
        )
        preprocessor = self.get_preprocessor(X_train_raw)

        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        return y_train, y_test, X_train, X_test

    def get_preprocessor(self, X):
        X = X.dropna()

        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return preprocessor
