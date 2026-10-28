# Logistics_Project
Python-based logistics data analysis project exploring retail sales, product demand, daily order patterns, and baseline demand forecasting.
# Strategic Planning and Data Exploration in Logistics

## Project Overview
This project uses Python to analyze retail transaction data and explore how data analysis can support logistics planning and business decision-making.

The project examines sales performance, product demand, daily order patterns, country-wise sales, and demand forecasting using a moving-average baseline.

## Objectives
- Clean and prepare retail transaction data.
- Analyze sales, quantities, and order patterns.
- Identify high-demand days and popular products.
- Compare sales performance across countries.
- Evaluate a baseline forecast for daily orders.
- Suggest improvements for inventory and operational planning.

## Tools and Technologies
- Python
- Pandas
- Matplotlib
- Jupyter Notebook or VS Code
- UCI Online Retail Dataset

## Project Structure
- `python_code/` — Python analysis script.
- `report/` — Final Word report.
- `data/` — Dataset files, if appropriate to share.
- `figures/` — Charts generated during the analysis.

## Key Findings
- Total sales value analyzed: £10,642,110.80.
- Total units sold: 5,572,420.
- Unique invoices: 19,960.
- Average daily orders: 65.44.
- Peak daily orders: 142.
- The United Kingdom accounted for approximately 84.59% of sales value by country.

## Forecasting
A moving-average baseline was used to forecast daily order volumes. The chronological test evaluation produced a Mean Absolute Error (MAE) of 29.54 orders and a Root Mean Squared Error (RMSE) of 37.70 orders.

## Dataset Source
UCI Machine Learning Repository — Online Retail Dataset:  
https://archive.ics.uci.edu/dataset/352/online+retail

## Project Deliverables
The repository contains the Python analysis code, project report, and visualizations.

## Future Improvements
Future work could explore regression models, clustering techniques, and route optimization if suitable additional data becomes available.
