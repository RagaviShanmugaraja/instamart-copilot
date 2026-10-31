
import pandas as pd

# Load cleaned data
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Convert order timestamp into datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep only completed orders
completed_orders = orders[
    orders["status"] == "completed"
].copy()

print("\nCompleted orders:", len(completed_orders))
print("Latest order date:", completed_orders["order_ts"].max())

# Display column names
print("CUSTOMER COLUMNS:")
print(customers.columns.tolist())

print("\nORDER COLUMNS:")
print(orders.columns.tolist())

# Display data types
print("\nORDER DATA TYPES:")
print(orders.dtypes)

# Display first five orders
print("\nFIRST FIVE ORDERS:")
print(orders.head())


# Reference date: one day after the latest order
snapshot_date = completed_orders["order_ts"].max() + pd.Timedelta(days=1)

# Calculate RFM values
rfm = completed_orders.groupby("customer_id").agg(
    Recency=("order_ts", lambda x: (snapshot_date - x.max()).days),
    Frequency=("order_id", "count"),
    Monetary=("order_amount", "sum")
).reset_index()

# Display first five customers
print("\nRFM ANALYSIS:")
print(rfm.head())

print("\nRFM SUMMARY:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# Save RFM results
rfm.to_csv("outputs/customer_rfm.csv", index=False)

print("\nRFM results saved successfully!")


# Calculate RFM scores from 1 to 5

rfm["R_score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Calculate combined RFM score
rfm["RFM_score"] = (
    rfm["R_score"].astype(str) +
    rfm["F_score"].astype(str) +
    rfm["M_score"].astype(str)
)

print("\nRFM SCORES:")
print(rfm.head(10))

# Save updated RFM results
rfm.to_csv("outputs/customer_rfm.csv", index=False)


# Assign customer segments based on RFM scores

def assign_segment(row):

    if row["R_score"] >= 4 and row["F_score"] >= 4 and row["M_score"] >= 4:
        return "Champions"

    elif row["F_score"] >= 4 and row["R_score"] >= 3:
        return "Loyal Customers"

    elif row["R_score"] >= 4 and row["F_score"] <= 2:
        return "Potential Loyalists"

    elif row["R_score"] <= 2 and row["F_score"] >= 3:
        return "At Risk"

    elif row["R_score"] <= 2 and row["F_score"] <= 2:
        return "Lost Customers"

    else:
        return "Others"


rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# Display segment counts
print("\nCUSTOMER SEGMENT DISTRIBUTION:")
print(rfm["Segment"].value_counts())

# Save updated RFM results
rfm.to_csv("outputs/customer_rfm.csv", index=False)

# Save customer segmentation results
rfm.to_csv("outputs/customer_segments.csv", index=False)

print("\nCustomer segmentation saved successfully!")