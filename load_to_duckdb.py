import duckdb
import pandas as pd
from pathlib import Path

CSV_PATH = Path("input.csv")
DB_PATH = Path("rag_elt.db")
TABLE_NAME = "raw_documents"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} が見つかりません。input.csv を用意してください。")

    df = pd.read_csv(CSV_PATH)
    if not {"id", "text"}.issubset(df.columns):
        raise ValueError("input.csv は少なくとも 'id', 'text' 列を含む必要があります。")

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        con.register("df_src", df)
        con.execute(
            f"""
            CREATE TABLE {TABLE_NAME} AS
            SELECT CAST(id AS BIGINT) AS id, CAST(text AS VARCHAR) AS text
            FROM df_src
            """
        )
        con.unregister("df_src")
    finally:
        con.close()
    print(f"Loaded {len(df)} rows into {DB_PATH}::{TABLE_NAME}")

if __name__ == "__main__":
    main()
