# ------------------------------------------------------------------------------
# Forecast expected return and covariance matrix using VAR + DCC-GARCH
#
# Usage:
#   Rscript var_dcc_garch_forecast.R <input_csv_path> [<output_dir>]
#
# Arguments:
#   input_csv_path: CSV file containing weekly log returns (row.names = date, columns = tickers)
#   output_dir: directory to write forecast outputs (default = data_processing/tmp)
#
# Output:
#   - expected_return.csv: vector of 1-step-ahead expected arithmetic returns (linear scale)
#   - forecast_cov_matrix.csv: 1-step-ahead forecasted covariance matrix
#
# Notes:
#   - This script fits a VAR model to forecast returns,
#     and a DCC-GARCH model to forecast the conditional covariance matrix.
#   - Expected returns are converted from log to linear using: exp(r) - 1
#   - All forecasting is done on weekly data (1-step ahead = 1 week)
# ------------------------------------------------------------------------------





library(tseries)
library(vars)
library(rmgarch)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
output_dir <- if (length(args) >= 2) args[2] else "."
log_returns <- read.csv(csv_path, row.names = 1)
log_returns <- ts(log_returns)


# fit VAR model
var_model <- vars::VAR(log_returns, type = "both", lag.max = 10, ic = "AIC")


fitted_values <- fitted(var_model)
residuals_df <- residuals(var_model)



# fit gcc-garch for covariance matrix

# univariate garch(1,1) spec
uspec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "norm"
)

# multivariate DCC spec
mspec <- dccspec(
  uspec = multispec(replicate(ncol(residuals_df), uspec)),
  dccOrder = c(1, 1),
  distribution = "mvnorm"
)

# fit residuals
dcc_garch_model <- dccfit(mspec, data = residuals_df)



# forecast

var_forecast <- predict(var_model, n.ahead = 1)
expected_return_log <- sapply(var_forecast$fcst, function(x) x[1, 1])
expected_return <- exp(expected_return_log) - 1

dcc_forecast <- dccforecast(dcc_garch_model, n.ahead = 1)
log_cov_matrix <- rcov(dcc_forecast)[[1]][, , 1]


write.csv(
  expected_return,
  file = file.path(output_dir, "expected_return.csv"),
  row.names = FALSE
)
write.csv(
  log_cov_matrix,
  file = file.path(output_dir, "forecast_cov_matrix.csv"),
  row.names = TRUE
)
