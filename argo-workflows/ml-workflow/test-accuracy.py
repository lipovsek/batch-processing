from sklearn.metrics import mean_squared_error
from joblib import load
import pandas as pd

from utils import get_boto_resource, create_engine, defaults


def test_accuracy():
    engine = create_engine()
    y_test, X_test = pd.read_sql_table("y_test", engine), pd.read_sql_table(
        "X_test", engine
    )
    model_pkl = "model.pkl"
    bucket, model_pth = "urly-bucket", f"data-mining/{model_pkl}"
    get_boto_resource().download_file(
        bucket, defaults["S3_path"], defaults["S3_bucket"]
    )
    result = load(model_pkl).predict(X_test)
    mean_squared_error(y_test, result)


if __name__ == "__main__":
    test_accuracy()
