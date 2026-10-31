import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("data/instamart.db")

# Read SQL queries
with open("sql/queries.sql", "r") as file:
    sql_content = file.read()

# Split queries using semicolon
queries = [
    query.strip()
    for query in sql_content.split(";")
    if query.strip()
]

# Run each query
for i, query in enumerate(queries, start=1):
    print(f"\nQUERY {i} RESULTS")
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    print("Number of rows:", len(df))

conn.close()