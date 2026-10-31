
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load raw customer data
customers = pd.read_csv(RAW_DIR / "customers.csv")

# 1. Standardise city names
customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

# 2. Standardise gender
customers["gender"] = (
    customers["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
)

gender_mapping = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female"
}

customers["gender"] = (
    customers["gender"]
    .map(gender_mapping)
    .fillna("unknown")
)

# 3. Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Treat ages outside 10-100 as missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = pd.NA

# Fill missing ages using the median age of each city
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# 4. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

# Save cleaned customer data
customers.to_csv(
    CLEAN_DIR / "customers.csv",
    index=False
)

# Display validation results
print("Cleaned customer dataset:", customers.shape)

print("\nCity counts:")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nData types:")
print(customers.dtypes)

print("\nCleaned customers.csv saved successfully!")
