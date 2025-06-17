import yfinance as yf

def fetch_price_data(tickers, period='1y'):
    """
    Download adjusted closing price data from Yahoo Finance.

    Parameters:
        tickers (list of str): List of stock tickers (e.g., ['AAPL', 'GOOGL'])
        period (str): Lookback period (e.g., '1y', '3mo', '5y')

    Returns:
        pd.DataFrame: DataFrame of adjusted closing prices indexed by date
    """
    data = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    return data.dropna()