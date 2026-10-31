
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

# Load all six CSV files
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
tickets = pd.read_csv(DATA_DIR / "tickets.csv")
policy_eval = pd.read_csv(DATA_DIR / "policy_eval.csv")
agent_scenarios = pd.read_csv(DATA_DIR / "agent_scenarios.csv")

# Explore each dataset
datasets = {
    "Customers": customers,
    "Products": products,
    "Orders": orders,
    "Tickets": tickets,
    "Policy Evaluation": policy_eval,
    "Agent Scenarios": agent_scenarios
}

for name, df in datasets.items():
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
