import pandas as pd


def create_df(data):
    records = []

    # Extract data from the defaultdict
    for key, metrics in data.items():
        dataset, algorithm, is_metacost = key.split("-")
        record = {
            "dataset": dataset,
            "algorithm": algorithm,
            "is_metacost": is_metacost,
            "sum": metrics.get("sum"),
            "f1": metrics.get("f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "time": metrics.get("time"),
        }
        records.append(record)

    # Create a DataFrame
    df = pd.DataFrame(records)
    return df
