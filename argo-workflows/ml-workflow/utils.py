# TODO improve secret management!
from sqlalchemy import create_engine as ce
import boto2
import os


def create_engine():
    user, password, database, hostname = (
        os.environ["DB_USER"],
        os.environ["DB_PASSWORD"],
        os.environ["DB_HOST"],
        os.environ["DB_NAME"],
    )
    return ce("postgresql+psycopg2://{user}:{password}@{hostname}/{database}")


def get_boto_resource():
    ACCESS_KEY, SECRET_KEY = os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_KEY"]
    return boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


defaults = {
    "cases": "cases",
    "X": "X",
    "y": "y",
    "X_test": "X_test",
    "y_test": "y_test",
    "S3_bucket": "urly-bucket",
    "S3_path": f"data-mining/model.pkl",
}
