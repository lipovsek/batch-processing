import pandas as pd
from utils import create_engine, defaults


def preprocess_data():
    csv_file = "cases.csv"
    df = pd.read_sql_table("cases", create_engine())
    df["week"] = df.date.dt.week
    df["year"] = df.date.dt.year
    weeks = df.groupby(["year", "week"]).date.nunique()
    valid = weeks[weeks == 7].reset_index()
    print(f"Full weeks calculated: {(len(valid)/len(weeks))*100}%")

    data = df[(df.week.isin(valid.week)) & (df.week.isin(valid.week))]
    X = data.groupby(["year", "week"])["cases.confirmed"].max().reset_index()
    y = data.groupby(["year", "week"])["cases.confirmed"].sum()
    engine = create_engine()
    X.to_sql(defaults["X"], engine)
    y.to_sql(defaults["y"], engine)


if __name__ == "__main__":
    preprocess_data()
