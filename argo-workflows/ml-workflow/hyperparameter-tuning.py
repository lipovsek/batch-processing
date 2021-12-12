from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

from utils import create_engine, get_boto_resource, defaults


def hyperparameter_tuning():
    """ engine = create_engine()
    X_train, X_test, y_train, y_test = train_test_split(
        pd.read_sql_table("X", engine),
        pd.read_sql_table("y", engine),
        test_size=0.10,
        random_state=42,
    )
    X_test.to_sql(defaults["X_test"], engine)
    y_test.to_sql(defaults["y_test"], engine)

    random_grid = {
        "max_depth": list(range(3, 51, 5))
        + [
            None,
        ],
        "n_estimators": list(range(5, 101, 10)),
        "min_samples_split": list(range(2, 5)),
        "min_samples_leaf": list(range(1, 5)),
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
    }
    rf_cv = RandomForestRegressor(random_state=30)
    rf_random = RandomizedSearchCV(
        estimator=rf_cv, param_distributions=random_grid, n_iter=200, cv=4, n_jobs=-1
    )
    rf_random.fit(X_train, y_train)
    model_pkl = "model.pkl"
    dump(rf_random.best_estimator_, model_pkl)
    get_boto_resource().upload_file(
        model_pkl, defaults["S3_bucket"], defaults["S3_path"]
    )
    print("Best parameters: ", rf_random.best_params_) """
    import os
    print("hyperparameter_tuning")
    print(os.environ)
    import time
    time.sleep(10)


if __name__ == "__main__":
    hyperparameter_tuning()
