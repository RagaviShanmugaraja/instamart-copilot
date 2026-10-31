
import pandas as pd
import sqlite3
from pathlib import Path

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

# Load cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# 1. Customer row count
assert len(customers) == 1200

# 2. Exactly six standardised cities
assert customers["city"].nunique() == 6

# 3. Expected city counts
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184
}
assert customers["city"].value_counts().to_dict() == expected_cities

# 4. Expected gender counts
expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30
}
assert customers["gender"].value_counts().to_dict() == expected_genders

# 5. No missing or invalid customer ages
assert customers["age"].notna().all()
assert customers["age"].between(18, 60).all()

# 6. Signup date is stored in the expected format
assert pd.to_datetime(customers["signup_date"], errors="coerce").notna().all()

# 7. Product count
assert len(products) == 40

# 8. Orders have one row per order ID
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309

# 9. Expected order statuses
assert orders["status"].value_counts().to_dict() == {
    "completed": 11110,
    "cancelled": 1199
}

# 10. No missing order amounts
assert orders["order_amount"].notna().all()

# 11. Completed orders have delivery times
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

assert completed["delivery_minutes"].notna().all()

# 12. Cancelled orders retain missing delivery times
assert cancelled["delivery_minutes"].isna().sum() == 1199

# 13. Completed order revenue
revenue = round(completed["order_amount"].sum(), 2)
assert revenue == 2998668.80

# 14. Verify SQLite database tables and row counts
with sqlite3.connect(DB_PATH) as conn:
    expected_counts = {
        "customers": 1200,
        "products": 40,
        "orders": 12309
    }

    for table, expected in expected_counts.items():
        actual = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            conn
        )["total"].iloc[0]

        assert actual == expected

print("ALL CHECKPOINTS PASSED")
