# TODO: improve secret management(would prefer to do it properly, rather than just as params)
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
import boto3


def fetch_data():
    from requests import get

    csv_file = "cases.csv"
    r = get(f"https://raw.githubusercontent.com/sledilnik/data/master/csv/{csv_file}")
    with open(csv_file, "w") as f:
        f.write(r.text)


def preprocess_data():
    import pandas as pd

    csv_file = "cases.csv"
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df["week"] = df.date.dt.week
    df["year"] = df.date.dt.year
    weeks = df.groupby(["year", "week"]).date.nunique()
    valid = weeks[weeks == 7].reset_index()
    print(f"Full weeks calculated: {(len(valid)/len(weeks))*100}%")

    data = df[(df.week.isin(valid.week)) & (df.week.isin(valid.week))]
    X = data.groupby(["year", "week"])["cases.confirmed"].max().reset_index()
    y = data.groupby(["year", "week"])["cases.confirmed"].sum()
    engine = create_engine()
    X.to_sql("X", engine)
    y.to_sql("y", engine)


def hyperparameter_tuning():
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from joblib import dump
    import pandas as pd

    engine = create_engine()
    X_train, X_test, y_train, y_test = train_test_split(
        pd.read_sql_table("X", engine),
        pd.read_sql_table("y", engine),
        test_size=0.10,
        random_state=42,
    )
    X_test.to_sql("X_test", engine)
    y_test.to_sql("y_test", engine)

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
    bucket, model_pth = "urly-bucket", f"data-mining/{model_pkl}"
    get_boto_resource().upload_file(model_pkl, bucket, model_pth)
    print("Best parameters: ", rf_random.best_params_)


def test_accuracy():
    from sklearn.metrics import mean_squared_error
    from joblib import load
    import pandas as pd

    y_test, X_test = pd.read_sql_table("y_test", engine), pd.read_sql_table(
        "X_test", engine
    )
    model_pkl = "model.pkl"
    bucket, model_pth = "urly-bucket", f"data-mining/{model_pkl}"
    get_boto_resource().download_file(bucket, model_pth, model_pkl)
    result = load(model_pkl).predict(X_test)
    mean_squared_error(y_test, result)


def create_engine():
    from sqlalchemy import create_engine as ce

    user, password, database, hostname = "user", "pass", "db", "localhost"
    return ce("postgresql+psycopg2://{user}:{password}@{hostname}/{database}")


def get_boto_resource():
    ACCESS_KEY, SECRET_KEY = "a", "b"
    return boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


df = ["SQLAlchemy==1.4.28", "psycopg==3.0.5", "pandas==1.3.4"]
s3 = [
    "boto3==1.20.23",
]
modeling = ["scikit-learn==1.0.1"]
with DAG(
    dag_id="example-ml-workflow",
    schedule_interval=None,
    start_date=None,
    catchup=False,
    tags=["example", "ml"],
) as dag:
    fetch_data_task = PythonVirtualenvOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
        requirements=[
            "requests==2.26.0",
        ]
        + df,
        system_site_packages=True,
        dag=dag,
    )
    preprocess_data_task = PythonVirtualenvOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        requirements=df,
        system_site_packages=True,
        dag=dag,
    )
    hyperparameter_tuning_task = PythonVirtualenvOperator(
        task_id="hyperparameter_tuning",
        python_callable=hyperparameter_tuning,
        requirements=modeling + df + s3,
        system_site_packages=True,
        dag=dag,
    )
    test_accuracy_task = PythonVirtualenvOperator(
        task_id="test_accuracy",
        python_callable=test_accuracy,
        requirements=modeling + df + s3,
        system_site_packages=True,
        dag=dag,
    )

    (
        fetch_data_task
        >> preprocess_data_task
        >> hyperparameter_tuning_task
        >> test_accuracy_task
    )
