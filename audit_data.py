
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

# Load raw datasets
customers = pd.read_csv(DATA_DIR / "customers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

# Keep empty ticket text as empty strings
tickets = pd.read_csv(DATA_DIR / "tickets.csv", keep_default_na=False)

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Tickets": tickets
}

# Shape, data types, missing counts and percentages
for name, df in datasets.items():
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    print("\nShape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    audit = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (df.isnull().sum() / len(df) * 100).round(2)
    })

    print("\nMissing value audit:")
    print(audit)

# Specific customer audit figures
print("\nCUSTOMER AUDIT")
print("Blank ages:", customers["age"].isna().sum())

age = pd.to_numeric(customers["age"], errors="coerce")
print("Ages outside 10-100:", ((age < 10) | (age > 100)).sum())

print("Distinct raw city spellings:", customers["city"].nunique())

gender = customers["gender"].dropna()
print("Distinct non-blank gender spellings:", gender.nunique())
print("Blank genders:", customers["gender"].isna().sum())

# Specific order audit figures
print("\nORDER AUDIT")
print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
print("Non-numeric order amounts:", amount.isna().sum())

delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Negative delivery times:", (delivery < 0).sum())
print("Blank delivery times:", delivery.isna().sum())
