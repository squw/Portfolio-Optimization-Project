from fetch_data import fetch_price_data
from data_processing.process_data_historical import process_historical_data
from data_processing.process_data_var_garch import process_data_var_garch
from optimize import optimize_portfolio

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
period = '3y'
method = 'pr3'
tau = 1.0

# Step 1: Fetch price data
price_data = fetch_price_data(tickers, period)

# Step 2: Process the data into returns and cov matrix
# expected_returns, cov_matrix, tickers = process_historical_data(price_data)

# Note: prodiction model predicts the value for a week after the presend day, then annualized
expected_returns, cov_matrix, tickers = process_data_var_garch(price_data)

# Step 3: Optimize
result = optimize_portfolio(expected_returns, cov_matrix, tickers, method, tau)

print(result)