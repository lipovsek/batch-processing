from requests import get
import pandas as pd

from utils import create_engine, defaults


def fetch_data():
    """ csv_file = "cases.csv"
    r = get(f"https://raw.githubusercontent.com/sledilnik/data/master/csv/{csv_file}")
    with open(csv_file, "w") as f:
        f.write(r.text)
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df.to_sql(defaults["cases"], create_engine()) """
    import os
    print("fetch_data")
    print(os.environ)
    import time
    time.sleep(10)


if __name__ == "__main__":
    fetch_data()
