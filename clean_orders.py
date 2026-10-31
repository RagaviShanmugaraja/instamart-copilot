
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load raw datasets
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

print("Initial orders:", len(orders))

# 1. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates()
print("Exact duplicate rows removed:", before - len(orders))

# 3. Keep the latest record for each order_id
before = len(orders)
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
)
print("Duplicate order records removed:", before - len(orders))

# 4. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Merge product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Fill missing amounts using product price, quantity and discount
missing_amounts = orders["order_amount"].isna().sum()

orders["order_amount"] = orders["order_amount"].fillna(
    (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
)

print("Missing amounts filled:", missing_amounts)

# 5. Set negative delivery times to missing
negative_times = (orders["delivery_minutes"] < 0).sum()

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

print("Negative delivery times corrected:", negative_times)

# Fill missing delivery times for completed orders
completed_mask = orders["status"] == "completed"

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform(lambda x: x.fillna(x.median()))
)

# Do not fill cancelled order delivery times

# Remove temporary product columns
orders = orders.drop(columns=["price_inr", "category"])

# Save cleaned orders
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# Validation results
print("\nCLEANED ORDERS SUMMARY")
print("Total rows:", len(orders))
print("\nStatus counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery times in completed orders:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery times in cancelled orders:",
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum()
)

print(
    "Completed order revenue:",
    round(
        orders.loc[orders["status"] == "completed", "order_amount"].sum(),
        2
    )
)

print("\nCleaned orders.csv saved successfully!")
