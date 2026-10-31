
import pandas as pd
import sqlite3
from pathlib import Path

CLEAN_DIR = Path("data/clean")
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/instamart.db")

# Load cleaned customers and orders
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Products have no cleaning issues, so copy the raw product data
products = pd.read_csv(RAW_DIR / "products.csv")

# Ensure order timestamps use the required text format
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Save all three clean CSV files
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# Create SQLite database and store the three tables
with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Verify table names and row counts
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )

    print("Database tables:")
    print(tables)

    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            conn
        )
        print(f"{table}: {count.iloc[0]['total']} rows")

print("\nSQLite database created successfully!")
