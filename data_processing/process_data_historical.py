import numpy as np

def process_historical_data(data):
    """
    Process raw price data into annualized expected returns and covariance matrix.

    Uses:
        - Monthly returns to estimate expected annual return
        - Daily returns to estimate annualized risk (covariance)

    Parameters:
        data (pd.DataFrame): Adjusted close price data from Yahoo Finance

    Returns:
        expected_annual_returns (np.ndarray): Annualized mean returns, shape (n_assets,)
        cov_matrix_annual (np.ndarray): Annualized covariance matrix, shape (n_assets, n_assets)
        tickers (list): List of ticker symbols
    """
    # Estimate annual return from monthly prices
    monthly_data = data.resample('ME').last()
    monthly_returns = monthly_data.pct_change().dropna()
    expected_annual_returns = monthly_returns.mean().to_numpy() * 12

    # Estimate annualized covariance from daily prices
    daily_returns = data.pct_change().dropna()
    cov_matrix_annual = daily_returns.cov().to_numpy() * 252

    tickers = list(data.columns)
    return expected_annual_returns, cov_matrix_annual, tickers