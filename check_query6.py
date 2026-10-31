import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

with open("sql/queries.sql", "r") as file:
    queries = [
        q.strip()
        for q in file.read().split(";")
        if q.strip()
    ]

df = pd.read_sql_query(queries[5], conn)

print("\nTOP CUSTOMER IN EACH CITY:")
print(
    df[df["spending_rank"] == 1][
        ["customer_id", "city", "total_spent"]
    ].to_string(index=False)
)

print("\nRANK DIFFERENCE CHECK:")
print(
    "Rows where RANK differs from ROW_NUMBER:",
    (df["spending_rank"] != df["row_number"]).sum()
)

print("\nTOTAL ROWS:", len(df))

conn.close()