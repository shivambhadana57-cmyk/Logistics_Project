import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the real retail dataset
file_path = "data/Online Retail.xlsx"

df = pd.read_excel(file_path)

# Display the first five rows
print(df.head())

print("\nDataset size:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Check data quality

print("\nDuplicate rows:", df.duplicated().sum())

print(
    "Cancellation invoices:",
    df["InvoiceNo"].astype(str).str.startswith("C").sum()
)

print("Zero or negative quantities:", (df["Quantity"] <= 0).sum())

print("Zero or negative prices:", (df["UnitPrice"] <= 0).sum())

# Create a separate copy for cleaning
clean_df = df.drop_duplicates().copy()

# Identify cancellation invoices
clean_df["IsCancellation"] = (
    clean_df["InvoiceNo"].astype(str).str.startswith("C")
)

# Create a sales-demand dataset
demand_df = clean_df[
    (~clean_df["IsCancellation"]) &
    (clean_df["Quantity"] > 0) &
    (clean_df["UnitPrice"] > 0)
].copy()

# Calculate total value for each transaction line
demand_df["TotalValue"] = (
    demand_df["Quantity"] * demand_df["UnitPrice"]
)

print("\nOriginal rows:", len(df))
print("Rows after duplicate removal:", len(clean_df))
print("Rows for demand analysis:", len(demand_df))

print("\nCleaned dataset preview:")
print(demand_df.head()) 


# Total number of units sold
total_units = demand_df["Quantity"].sum()

# Total sales value
total_sales = demand_df["TotalValue"].sum()

# Number of unique invoices
total_orders = demand_df["InvoiceNo"].nunique()

print("\n--- Logistics Analysis Summary ---")
print("Total units sold:", total_units)
print("Total sales value:", round(total_sales, 2))
print("Unique invoices:", total_orders)

# Top 10 products by quantity sold
top_products = (
    demand_df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 products by quantity sold:")
print(top_products)

# Visualize the 10 most-sold products
top_products.sort_values().plot(
    kind="barh",
    figsize=(10, 6),
    title="Top 10 Products by Quantity Sold"
)

plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


# STEP 8: Daily Demand Analysis

# Extract the date from each invoice timestamp
demand_df["Date"] = demand_df["InvoiceDate"].dt.date

# Summarize demand for each day
daily_demand = (
    demand_df.groupby("Date")
    .agg(
        Total_Orders=("InvoiceNo", "nunique"),
        Total_Units=("Quantity", "sum"),
        Total_Sales=("TotalValue", "sum")
    )
    .reset_index()
)

print("\n--- Daily Demand Summary ---")
print(daily_demand.head(10))

# Find the busiest day by order count
busiest_day = daily_demand.loc[
    daily_demand["Total_Orders"].idxmax()
]

print("\nBusiest day by number of orders:")
print(busiest_day)

# Find the day with the most units sold
highest_units_day = daily_demand.loc[
    daily_demand["Total_Units"].idxmax()
]

print("\nDay with the highest quantity sold:")
print(highest_units_day)


# Plot daily order volume

plt.figure(figsize=(12, 5))

plt.plot(
    pd.to_datetime(daily_demand["Date"]),
    daily_demand["Total_Orders"]
)

plt.title("Daily Order Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Analyze orders by weekday

demand_df["Weekday"] = (
    demand_df["InvoiceDate"].dt.day_name()
)

weekday_demand = (
    demand_df.groupby("Weekday")
    .agg(
        Total_Orders=("InvoiceNo", "nunique"),
        Total_Units=("Quantity", "sum")
    )
    .reindex([
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
)

print("\n--- Demand by Weekday ---")
print(weekday_demand)

weekday_demand["Total_Orders"].plot(
    kind="bar",
    figsize=(9, 5),
    title="Total Orders by Weekday"
)

plt.xlabel("Weekday")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 9: Calculate KPIs

kpis = {
    "Total Orders": daily_demand["Total_Orders"].sum(),
    "Total Units Sold": daily_demand["Total_Units"].sum(),
    "Average Daily Orders": daily_demand["Total_Orders"].mean(),
    "Average Daily Sales": daily_demand["Total_Sales"].mean(),
    "Peak Day Orders": daily_demand["Total_Orders"].max()
}

print("\n--- PROJECT KPI REPORT ---")

for name, value in kpis.items():
    print(f"{name}: {value:,.2f}")



# STEP 10: Forecast Daily Orders

# Prepare the daily order data
forecast_df = daily_demand[["Date", "Total_Orders"]].copy()

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df = forecast_df.sort_values("Date")

# Fill missing calendar dates with zero orders
forecast_df = (
    forecast_df.set_index("Date")
    .asfreq("D", fill_value=0)
    .reset_index()
)

# Create a simple 7-day moving-average forecast
forecast_df["Forecast_Orders"] = (
    forecast_df["Total_Orders"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

print("\n--- Actual Orders and Forecast ---")
print(forecast_df.tail(15))

# Plot actual orders and forecast
plt.figure(figsize=(12, 5))

plt.plot(
    forecast_df["Date"],
    forecast_df["Total_Orders"],
    label="Actual Orders"
)

plt.plot(
    forecast_df["Date"],
    forecast_df["Forecast_Orders"],
    label="7-Day Forecast"
)

plt.title("Daily Orders: Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# STEP 11: Evaluate Forecast Accuracy

# Keep only dates where a forecast is available
evaluation_df = forecast_df.dropna(
    subset=["Forecast_Orders"]
).copy()

# Calculate prediction errors
evaluation_df["Error"] = (
    evaluation_df["Total_Orders"]
    - evaluation_df["Forecast_Orders"]
)

evaluation_df["Absolute_Error"] = (
    evaluation_df["Error"].abs()
)

# Mean Absolute Error (MAE)
mae = evaluation_df["Absolute_Error"].mean()

# Root Mean Squared Error (RMSE)
rmse = (
    evaluation_df["Error"].pow(2).mean()
) ** 0.5

print("\n--- Forecast Accuracy ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print("\n--- Sample Forecast Errors ---")
print(
    evaluation_df[
        ["Date", "Total_Orders", "Forecast_Orders", "Error"]
    ].tail(10).round(2)
)



# STEP 11: Evaluate Forecast Accuracy

# Keep only dates where a forecast is available
evaluation_df = forecast_df.dropna(
    subset=["Forecast_Orders"]
).copy()

# Calculate prediction errors
evaluation_df["Error"] = (
    evaluation_df["Total_Orders"]
    - evaluation_df["Forecast_Orders"]
)

evaluation_df["Absolute_Error"] = (
    evaluation_df["Error"].abs()
)

# Mean Absolute Error (MAE)
mae = evaluation_df["Absolute_Error"].mean()

# Root Mean Squared Error (RMSE)
rmse = (
    evaluation_df["Error"].pow(2).mean()
) ** 0.5

print("\n--- Forecast Accuracy ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print("\n--- Sample Forecast Errors ---")
print(
    evaluation_df[
        ["Date", "Total_Orders", "Forecast_Orders", "Error"]
    ].tail(10).round(2)
)



# STEP 12: Time-Based Train/Test Evaluation

# Prepare the daily series in chronological order
model_df = forecast_df[
    ["Date", "Total_Orders"]
].dropna().copy()

model_df = model_df.sort_values("Date").reset_index(drop=True)

# Reserve the latest 20% of dates for testing
split_point = int(len(model_df) * 0.8)

train_df = model_df.iloc[:split_point].copy()
test_df = model_df.iloc[split_point:].copy()

# Baseline: predict each test day's orders
# using the average orders from the previous 7 days.
history = list(train_df["Total_Orders"])
predictions = []

for actual in test_df["Total_Orders"]:
    prediction = sum(history[-7:]) / min(len(history), 7)
    predictions.append(prediction)

    # Add the actual observed day to history before forecasting
    # the next day. This simulates updating the forecast daily.
    history.append(actual)

test_df["Forecast_Orders"] = predictions

# Calculate test-period errors
test_df["Error"] = (
    test_df["Total_Orders"] - test_df["Forecast_Orders"]
)

test_mae = test_df["Error"].abs().mean()
test_rmse = (test_df["Error"].pow(2).mean()) ** 0.5

print("\n--- Time-Based Test Results ---")
print(f"Training days: {len(train_df)}")
print(f"Testing days: {len(test_df)}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

print("\n--- First 10 Test Predictions ---")
print(
    test_df[
        ["Date", "Total_Orders", "Forecast_Orders", "Error"]
    ].head(10).round(2).to_string(index=False)
)

# Plot actual versus forecast in the test period
plt.figure(figsize=(12, 5))
plt.plot(
    test_df["Date"], test_df["Total_Orders"],
    label="Actual Orders"
)
plt.plot(
    test_df["Date"], test_df["Forecast_Orders"],
    label="Forecast Orders"
)
plt.title("Test Period: Actual vs Forecast Daily Orders")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# STEP 13: Identify High-Demand Days

# Calculate the 90th percentile of daily order volume
high_demand_threshold = daily_demand["Total_Orders"].quantile(0.90)

# Identify days at or above this threshold
high_demand_days = daily_demand[
    daily_demand["Total_Orders"] >= high_demand_threshold
].copy()

high_demand_days = high_demand_days.sort_values(
    "Total_Orders", ascending=False
)

print("\n--- High-Demand Analysis ---")
print(f"90th percentile threshold: {high_demand_threshold:.2f} orders")
print(f"Number of high-demand days: {len(high_demand_days)}")

print("\nTop 10 High-Demand Days:")
print(
    high_demand_days[
        ["Date", "Total_Orders", "Total_Units", "Total_Sales"]
    ].head(10).to_string(index=False)
)

# Plot the distribution of daily orders
plt.figure(figsize=(10, 5))
plt.hist(daily_demand["Total_Orders"], bins=20, edgecolor="black")
plt.axvline(
    high_demand_threshold,
    linestyle="--",
    label="90th Percentile Threshold"
)

plt.title("Distribution of Daily Order Volume")
plt.xlabel("Orders per Day")
plt.ylabel("Number of Days")
plt.legend()
plt.tight_layout()
plt.show()



# STEP 14: Country and Product Analysis

# Top 10 countries by sales value
country_sales = (
    demand_df.groupby("Country")
    .agg(
        Total_Sales=("TotalValue", "sum"),
        Total_Orders=("InvoiceNo", "nunique"),
        Total_Units=("Quantity", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)

print("\n--- Top 10 Countries by Sales ---")
print(country_sales.head(10).round(2).to_string())

# Top 10 products by sales value
product_sales = (
    demand_df.groupby("Description")
    .agg(
        Total_Sales=("TotalValue", "sum"),
        Total_Units=("Quantity", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)

print("\n--- Top 10 Products by Sales Value ---")
print(product_sales.head(10).round(2).to_string())

# Plot top 10 countries by sales
country_sales.head(10)["Total_Sales"].sort_values().plot(
    kind="barh",
    figsize=(10, 6),
    title="Top 10 Countries by Sales Value"
)

plt.xlabel("Total Sales Value")
plt.ylabel("Country")
plt.tight_layout()
plt.show()



# STEP 15: Final Project Summary

# Calculate the percentage of sales from the United Kingdom
uk_sales = country_sales.loc["United Kingdom", "Total_Sales"]
all_country_sales = country_sales["Total_Sales"].sum()
uk_sales_share = (uk_sales / all_country_sales) * 100

# Identify the top country and product
top_country = country_sales["Total_Sales"].idxmax()
top_product = product_sales["Total_Sales"].idxmax()

# Create a summary
print("\n" + "=" * 45)
print("FINAL LOGISTICS ANALYSIS SUMMARY")
print("=" * 45)

print(f"Total sales value: £{demand_df['TotalValue'].sum():,.2f}")
print(f"Total units sold: {demand_df['Quantity'].sum():,}")
print(f"Total unique invoices: {demand_df['InvoiceNo'].nunique():,}")
print(f"Average daily orders: {daily_demand['Total_Orders'].mean():.2f}")
print(f"Peak daily orders: {daily_demand['Total_Orders'].max():.0f}")
print(f"High-demand days: {len(high_demand_days)}")
print(f"High-demand threshold: {high_demand_threshold:.2f} orders")
print(f"Test forecast MAE: {test_mae:.2f} orders")
print(f"Test forecast RMSE: {test_rmse:.2f} orders")
print(f"Top country by sales: {top_country}")
print(f"UK share of sales: {uk_sales_share:.2f}%")
print(f"Top product by sales value: {top_product}")

print("=" * 45)
