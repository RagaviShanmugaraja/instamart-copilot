from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib

# Define cutoff date
CUTOFF_DATE = pd.Timestamp("2025-09-01")

# Load cleaned customer and order data
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Separate completed orders and final-status orders
completed_orders = orders[
    orders["status"] == "completed"
].copy()

final_orders = orders[
    orders["status"].isin(["completed", "cancelled"])
].copy()

# Use only orders before the cutoff
historical_completed = completed_orders[
    completed_orders["order_ts"] < CUTOFF_DATE
].copy()

historical_final = final_orders[
    final_orders["order_ts"] < CUTOFF_DATE
].copy()

print("CUSTOMERS:", len(customers))
print("HISTORICAL COMPLETED ORDERS:", len(historical_completed))
print("HISTORICAL FINAL ORDERS:", len(historical_final))

# Identify completed orders during September 2025
september_completed = completed_orders[
    (completed_orders["order_ts"] >= "2025-09-01") &
    (completed_orders["order_ts"] < "2025-10-01")
].copy()

# Get customers who ordered during September
active_september_customers = set(
    september_completed["customer_id"]
)

# Build cohort: customers with at least one completed
# order before the cutoff
cohort_customers = set(
    historical_completed["customer_id"]
)

# Create customer cohort DataFrame
cohort = customers[
    customers["customer_id"].isin(cohort_customers)
].copy()

# Create lapse label
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_september_customers)
).astype(int)

# Display results
print("\nLAPSE LABEL DISTRIBUTION:")
print(cohort["lapsed"].value_counts())

print("\nTOTAL COHORT:", len(cohort))
print("LAPSED CUSTOMERS:", cohort["lapsed"].sum())

print(
    "LAPSED PERCENTAGE:",
    round(cohort["lapsed"].mean() * 100, 2),
    "%"
)

# Load product data
products = pd.read_csv("data/clean/products.csv")

print("\nPRODUCT COLUMNS:")
print(products.columns.tolist())

print("\nFIRST FIVE PRODUCTS:")
print(products.head())


# Merge historical completed orders with product categories
historical_completed = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Calculate purchasing features
customer_features = historical_completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (CUTOFF_DATE - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

print("\nHISTORICAL CUSTOMER FEATURES:")
print(customer_features.head())

print("\nFEATURE SUMMARY:")
print(customer_features.describe().round(2))

# Calculate cancellation rate using historical final-status orders
cancellation_rate = (
    historical_final.groupby("customer_id")
    .agg(
        total_final_orders=("order_id", "count"),
        cancelled_orders=(
            "status",
            lambda x: (x == "cancelled").sum()
        )
    )
    .reset_index()
)

cancellation_rate["cancellation_rate"] = (
    cancellation_rate["cancelled_orders"] /
    cancellation_rate["total_final_orders"]
)

# Merge cancellation rate into customer features
customer_features = customer_features.merge(
    cancellation_rate[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)

print("\nCANCELLATION RATE:")
print(customer_features[
    ["customer_id", "cancellation_rate"]
].head())

print("\nAVERAGE CANCELLATION RATE:")
print(customer_features["cancellation_rate"].mean())


# Calculate most frequently ordered product category
favourite_category = (
    historical_completed
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
)

# Select the most frequently ordered category per customer
favourite_category = (
    favourite_category
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
)

favourite_category = favourite_category[
    ["customer_id", "category"]
].rename(columns={"category": "favourite_category"})

# Merge favourite category into customer features
customer_features = customer_features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

print("\nFAVOURITE PRODUCT CATEGORY:")
print(
    customer_features[
        ["customer_id", "favourite_category"]
    ].head(10)
)

print("\nCATEGORY DISTRIBUTION:")
print(customer_features["favourite_category"].value_counts())

# Calculate customer tenure
customer_profile = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_profile["tenure_days"] = (
    CUTOFF_DATE - customer_profile["signup_date"]
).dt.days

# Remove signup_date because tenure_days represents it
customer_profile = customer_profile.drop(
    columns=["signup_date"]
)

# Merge customer profile into historical features
customer_features = customer_features.merge(
    customer_profile,
    on="customer_id",
    how="left"
)

print("\nCUSTOMER PROFILE FEATURES:")
print(customer_features[
    ["customer_id", "city", "age", "gender", "tenure_days"]
].head())

print("\nCUSTOMER FEATURE COLUMNS:")
print(customer_features.columns.tolist())

# Merge historical features with lapse labels
model_data = customer_features.merge(
    cohort[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner"
)

# Check dataset shape
print("\nMODEL DATA SHAPE:")
print(model_data.shape)

# Check missing values
print("\nMISSING VALUES:")
print(model_data.isnull().sum())

# Check target distribution
print("\nLAPSE LABEL DISTRIBUTION:")
print(model_data["lapsed"].value_counts())

# Separate features and target

X = model_data.drop(
    columns=["customer_id", "lapsed"]
)

y = model_data["lapsed"]

print("\nFEATURES (X):")
print(X.head())

print("\nTARGET (y):")
print(y.head())

print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAINING DATA SHAPE:", X_train.shape)
print("TESTING DATA SHAPE:", X_test.shape)

print("\nTRAINING LABEL DISTRIBUTION:")
print(y_train.value_counts())

print("\nTESTING LABEL DISTRIBUTION:")
print(y_test.value_counts())

# Identify numerical and categorical columns

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Numerical preprocessing
numeric_transformer = StandardScaler()

# Categorical preprocessing
categorical_transformer = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting again
X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSING RESULTS:")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\nLEARNED FREQUENCY MEAN:")
print(
    preprocessor.named_transformers_["num"].mean_[1]
)

print("\nFULL COHORT FREQUENCY MEAN:")
print(model_data["frequency"].mean())

# Confirm no missing values in original features
print("\nMISSING VALUES IN FEATURES:")
print(X.isnull().sum().sum())

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

# Create baseline model
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

# Predict on test data
baseline_predictions = baseline.predict(X_test)

# Evaluate baseline
print("\nBASELINE MODEL RESULTS:")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))

print(
    "Lapsed-class F1:",
    f1_score(y_test, baseline_predictions, zero_division=0)
)

# Logistic Regression pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Hyperparameter values
logistic_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

# GridSearchCV
logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
logistic_grid.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION RESULTS:")
print("Best parameters:", logistic_grid.best_params_)
print("Best cross-validation F1:", logistic_grid.best_score_)

# Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Hyperparameter grid
rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

# GridSearchCV
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
rf_grid.fit(X_train, y_train)

print("\nRANDOM FOREST RESULTS:")
print("Best parameters:", rf_grid.best_params_)
print("Best cross-validation F1:", rf_grid.best_score_)

# Get best tuned models
best_logistic = logistic_grid.best_estimator_
best_rf = rf_grid.best_estimator_

# Make test predictions
logistic_predictions = best_logistic.predict(X_test)
rf_predictions = best_rf.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, predictions):

    print(f"\n{name} TEST RESULTS")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    print(
        "Precision:",
        precision_score(y_true, predictions, zero_division=0)
    )

    print(
        "Recall:",
        recall_score(y_true, predictions, zero_division=0)
    )

    print(
        "F1 Score:",
        f1_score(y_true, predictions, zero_division=0)
    )


# Evaluate both models
evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    logistic_predictions
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)

# CLASS-BALANCING EXPERIMENT

# Combine training features and target
train_df = X_train.copy()
train_df["lapsed"] = y_train.values

# Separate the two classes
not_lapsed = train_df[train_df["lapsed"] == 0]
lapsed = train_df[train_df["lapsed"] == 1]

# Duplicate lapsed customers until both classes have equal rows
lapsed_oversampled = lapsed.sample(
    n=len(not_lapsed) - len(lapsed),
    replace=True,
    random_state=42
)

# Combine original rows and duplicated rows
balanced_train = pd.concat(
    [not_lapsed, lapsed, lapsed_oversampled],
    ignore_index=True
)

# Separate features and target again
X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("\nBALANCED TRAINING DATA")
print(y_train_balanced.value_counts())
print("Total rows:", len(y_train_balanced))

# Create a fresh copy of the best Logistic Regression pipeline
balanced_logistic = clone(logistic_grid.best_estimator_)

# Train using balanced data
balanced_logistic.fit(X_train_balanced, y_train_balanced)

# Evaluate on the original, unchanged test data
balanced_predictions = balanced_logistic.predict(X_test)

print("\nBALANCED LOGISTIC REGRESSION RESULTS")
print("Confusion Matrix:")
print(confusion_matrix(y_test, balanced_predictions))

print("Precision:", precision_score(
    y_test, balanced_predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, balanced_predictions, zero_division=0
))

# FEATURE IMPORTANCE - RANDOM FOREST

best_rf = rf_grid.best_estimator_

rf_classifier = best_rf.named_steps["classifier"]
rf_preprocessor = best_rf.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance_df.head(5).to_string(index=False))


# LOGISTIC REGRESSION COEFFICIENTS

best_logistic = logistic_grid.best_estimator_

logistic_classifier = best_logistic.named_steps["classifier"]
logistic_preprocessor = best_logistic.named_steps["preprocessor"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_classifier.coef_[0]
})

selected_features = [
    "num__recency",
    "num__frequency",
    "num__cancellation_rate"
]

print("\nLOGISTIC REGRESSION COEFFICIENTS")

for feature in selected_features:
    result = coefficient_df[
        coefficient_df["feature"] == feature
    ]

    print(result.to_string(index=False))

    # SAVE THE SELECTED RANDOM FOREST MODEL

selected_model = rf_grid.best_estimator_

joblib.dump(
    selected_model,
    "outputs/lapse_model.joblib"
)

print("\nMODEL SAVED SUCCESSFULLY")


# RELOAD THE MODEL
reloaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

# Compare predictions before and after reloading
original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

print(
    "Predictions identical after reload:",
    np.array_equal(original_predictions, reloaded_predictions)
)

segments = pd.read_csv("outputs/customer_segments.csv")

print("\nSEGMENT FILE COLUMNS:")
print(segments.columns.tolist())


# RETENTION LIST - PREMIUM CUSTOMERS

# Load the corrected K-means segment file
segments = pd.read_csv("outputs/customer_segments.csv")

# Get customer IDs from the untouched test set
test_customer_ids = model_data.loc[
    X_test.index, "customer_id"
].values

# Predict lapse probabilities using the saved Random Forest model
lapse_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

# Create test customer predictions
test_results = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.values
})

# Identify Premium customers
premium_customers = segments[
    segments["is_premium"] == True
][["customer_id"]]

# Keep only Premium customers who are in the test set
premium_test_results = test_results.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)

# Rank customers by predicted lapse probability
premium_test_results = premium_test_results.sort_values(
    by="lapse_probability",
    ascending=False
)

# Select top 10 customers (or all if fewer than 10)
retention_list = premium_test_results.head(10)

# Save required columns
retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nRETENTION LIST:")
print(retention_list.to_string(index=False))

print("\nTOTAL PREMIUM CUSTOMERS IN TEST SET:",
      len(premium_test_results))

print("CUSTOMERS IN RETENTION LIST:",
      len(retention_list))

print("TRULY LAPSED CUSTOMERS IN RETENTION LIST:",
      retention_list["actually_lapsed"].sum())

print("\nRetention list saved successfully!")


clusters = pd.read_csv("outputs/customer_clusters.csv")
summary = pd.read_csv("outputs/cluster_summary.csv")

print("\nCUSTOMER CLUSTERS COLUMNS:")
print(clusters.columns.tolist())
print(clusters.head())

print("\nCLUSTER SUMMARY COLUMNS:")
print(summary.columns.tolist())
print(summary.to_string(index=False))