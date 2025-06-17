import numpy as np
import cvxpy as cp

def optimize_portfolio(expected_returns, cov_matrix, tickers, method='pr1', tau=1.0, target_return=None):
    """
    Optimize portfolio weights based on one of three risk-return models.

    Parameters:
        expected_returns (np.ndarray): Estimated annual returns for each asset (shape: n_assets,)
        cov_matrix (np.ndarray): Annualized covariance matrix (shape: n_assets x n_assets)
        tickers (list of str): List of ticker symbols (used to label output)
        method (str): One of 'pr1', 'pr2', or 'pr3'
            - 'pr1': Minimize variance (risk)
            - 'pr2': Minimize variance with a minimum expected return constraint
            - 'pr3': Maximize risk-adjusted return (quadratic utility function)
        tau (float): Risk aversion parameter (only used in 'pr3', default = 1.0)
        target_return (float or None): Minimum required annual return (only used in 'pr2')

    Returns:
        dict: {
            'weights': {ticker: weight},
            'expected_return': float,
            'volatility': float,
            'sharpe_ratio': float
        }
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)

    port_return = expected_returns @ weights
    port_variance = cp.quad_form(weights, cov_matrix)

    constraints = [cp.sum(weights) == 1, weights >= 0]

    if method == 'pr1':
        objective = cp.Minimize(port_variance)
    elif method == 'pr2':
        if target_return is None:
            raise ValueError("PR2 requires a target_return argument.")
        constraints.append(port_return >= target_return)
        objective = cp.Minimize(port_variance)
    elif method == 'pr3':
        objective = cp.Maximize(port_return - (tau / 2) * port_variance)
    else:
        raise ValueError("Invalid method. Choose 'pr1', 'pr2', or 'pr3'.")

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL or weights.value is None:
        raise RuntimeError(f"Optimization failed: {problem.status}")

    opt_weights = weights.value
    port_volatility = np.sqrt(opt_weights.T @ cov_matrix @ opt_weights)
    port_return_value = expected_returns @ opt_weights
    sharpe_ratio = port_return_value / port_volatility if port_volatility > 0 else 0

    return {
        'weights': dict(zip(tickers, np.round(opt_weights, 4))),
        'expected_return': round(port_return_value, 4),
        'volatility': round(port_volatility, 4),
        'sharpe_ratio': round(sharpe_ratio, 4)
    }