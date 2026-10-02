# Car Resale Price Intelligence

Used-car resale analysis and Streamlit dashboard based on the supplied `public_cars.csv`. The project cleans the data, preserves the original USD target, converts prices using the documented rate `1 USD = ₹83.50`, performs Plotly EDA, and compares Linear Regression, Random Forest, and Gradient Boosting with MAE, RMSE, and R².

`feature_0`–`feature_9` are renamed `Feature_01`–`Feature_10` because no supplied metadata establishes their meanings. Target-derived columns are excluded from ML predictors. Exact duplicates are removed; numeric missing values use median imputation, categorical missing values use `unknown`, and boolean missing values use `False`. Outliers are not blindly deleted.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
`data/public_cars.csv` is the source copy; `data/cleaned_cars.csv` is the cleaned dataset; `analysis.py` contains reusable functions; `app.py` is the dashboard.
