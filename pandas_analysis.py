import os
import pandas as pd
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Select completed orders
completed_orders = orders[
    orders["status"] == "completed"
]

# Merge orders with customers
sales = completed_orders.merge(
    customers,
    on="customer_id",
    how="left"
)

# Merge with products
sales = sales.merge(
    products,
    on="product_id",
    how="left"
)

# Calculate revenue by city and category
revenue_summary = pd.pivot_table(
    sales,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

revenue_summary.to_csv("outputs/revenue_by_city_category.csv")

print("\nREVENUE BY CITY AND CATEGORY")
print(revenue_summary)

print("\nGRAND TOTAL REVENUE:")
print(sales["order_amount"].sum())

# City-level sales performance

city_performance = (
    sales.groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        order_count=("order_id", "count")
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)

city_performance.to_csv(
    "outputs/city_sales_performance.csv",
    index=False
)

print("\nCITY-LEVEL SALES PERFORMANCE")
print(city_performance.to_string(index=False))

# Order amount distribution and outlier analysis

amount = sales["order_amount"]

Q1 = amount.quantile(0.25)
Q3 = amount.quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = amount[
    (amount < lower_limit) | (amount > upper_limit)
]

print("\nORDER AMOUNT DISTRIBUTION")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Standard Deviation:", round(amount.std(), 2))
print("Skewness:", round(amount.skew(), 2))
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower Limit:", round(lower_limit, 2))
print("Upper Limit:", round(upper_limit, 2))
print("Total Outliers:", len(outliers))
print("Lower Outliers:", (amount < lower_limit).sum())
print("Upper Outliers:", (amount > upper_limit).sum())

distribution_summary = pd.DataFrame({
    "Metric": [
        "Mean",
        "Median",
        "Standard Deviation",
        "Skewness",
        "Q1",
        "Q3",
        "IQR",
        "Lower Limit",
        "Upper Limit",
        "Total Outliers",
        "Lower Outliers",
        "Upper Outliers"
    ],
    "Value": [
        amount.mean(),
        amount.median(),
        amount.std(),
        amount.skew(),
        Q1,
        Q3,
        IQR,
        lower_limit,
        upper_limit,
        len(outliers),
        (amount < lower_limit).sum(),
        (amount > upper_limit).sum()
    ]
})

distribution_summary.to_csv(
    "outputs/order_amount_distribution.csv",
    index=False
)

# Correlation analysis

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = sales[correlation_columns].corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(3))

# Find the strongest correlation
correlation_pairs = (
    correlation_matrix
    .where(
        ~pd.DataFrame(
            [[i == j for j in range(len(correlation_columns))]
             for i in range(len(correlation_columns))],
            index=correlation_columns,
            columns=correlation_columns
        )
    )
    .stack()
)

print("\nSTRONGEST CORRELATION:")
print(correlation_pairs.abs().idxmax())
print("Correlation value:", correlation_pairs.loc[correlation_pairs.abs().idxmax()])

correlation_matrix.to_csv("outputs/correlation_matrix.csv")