import numpy as np
import pandas as pd
import subprocess
import os

def process_data_var_garch(data):
    
    """
    Process price data and forecast next-period expected return and covariance matrix using VAR + DCC-GARCH (via R script).

    Usage:
        process_data_var_garch(data: pd.DataFrame) -> (np.ndarray, np.ndarray, list)

    Parameters:
        data (pd.DataFrame): Raw adjusted close price data indexed by date and with asset tickers as columns.
                            Data should be at daily resolution or finer.

    Steps:
        1. Resample data to weekly frequency and compute log returns.
        2. Write log returns to a temporary CSV file.
        3. Call external R script (var_dcc_garch_forecast.R) to:
            - Fit a VAR model to forecast 1-step-ahead returns.
            - Fit a DCC-GARCH model to forecast the conditional covariance matrix.
        4. Read back forecasted expected return and covariance matrix from R output.
        5. Annualize both outputs (assuming 52 trading weeks per year).

    Returns:
        expected_return_annual (np.ndarray): Annualized expected arithmetic returns, shape (n_assets,)
        cov_matrix_annual (np.ndarray): Annualized forecasted covariance matrix, shape (n_assets, n_assets)
        tickers (list): List of asset ticker strings

    Note:
        - Assumes the R script is located at data_processing/fun/var_dcc_garch_forecast.R.
        - R script output files must include: expected_return.csv and forecast_cov_matrix.csv.
        - Intermediate files are stored under data_processing/fun/tmp/.
    """

    
    
    weekly_data = data.resample('W-FRI').last()
    weekly_return_log = np.log(weekly_data).diff().dropna()
    
    # creates tmp csv file for r
    base_dir = os.path.dirname(__file__)
    print("base dir = ", base_dir)
    tmp_dir = os.path.join(base_dir, "tmp")
    print("tmp dir = ", tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    csv_path = os.path.join(tmp_dir, "weekly_return_log.csv")
    print("csv path = ", csv_path)
    weekly_return_log.to_csv(csv_path)
    
    r_path = os.path.join(base_dir, "fun/var_dcc_garch_forecast.R")
    subprocess.run(["Rscript", r_path, csv_path, tmp_dir], check=True)
    
    expected_return_path = os.path.join(tmp_dir, "expected_return.csv")
    cov_matrix_path = os.path.join(tmp_dir, "forecast_cov_matrix.csv")
    expected_return_df = pd.read_csv(expected_return_path)
    cov_matrix_df = pd.read_csv(cov_matrix_path, index_col = 0)
    
    expected_returns = expected_return_df.to_numpy().flatten()
    cov_matrix = cov_matrix_df.to_numpy()
    
    expected_return_annual = expected_returns * 52
    cov_matrix_annual = cov_matrix * 52
    
    tickers = list(data.columns)
    
    return expected_return_annual, cov_matrix_annual, tickers