# Time Series Forecasting with AutoARIMA

This Python script performs time series forecasting using the AutoARIMA model. It reads time series data from a file, preprocesses it, fits the AutoARIMA model to the data, makes forecasts, and evaluates the model's performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## Usage

1. **Data Preparation**: Ensure your time series data is stored in a file. The file should be in CSV format with two columns: 'variable' (containing date/time values) and 'value' (containing the time series data).

2. **Install Dependencies**: Make sure you have the necessary Python packages installed. You can install them using pip:

    ```
    pip install pandas numpy seaborn scipy matplotlib statsmodels
    ```

3. **Run the Script**: Execute the Python script `auto_arima_forecast.py`. Make sure to provide the correct file path to your data file. You can do this by editing the `file_path` variable in the script.

4. **Interpret Results**: The script will display various plots, including raw data, forecasts, and residual analysis. Additionally, it will print the RMSE and MAE values, providing insights into the model's accuracy.

## Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- scipy
- matplotlib
- statsmodels

## File Structure

- `auto_arima_forecast.py`: Main Python script for time series forecasting.
- `README.md`: This README file providing instructions and information about the script.
- `data.csv`: Sample time series data file (replace this with your own data file).

