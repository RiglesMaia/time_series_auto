#R.Maia
#Auto Time Series
###################

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from statsforecast import StatsForecast as sf
from statsforecast.models import AutoARIMA as aa
from statsmodels.graphics.tsaplots import plot_acf

# Function to read data from a file
def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

# Function to preprocess the data
def preprocess_data(paddock):
    paddock['variable'] = pd.to_datetime(paddock['variable'])  # Convert to datetime
    paddock = paddock.rename(columns={'variable': 'data', 'value': 'NDVI'})  # Rename columns
    return paddock

# Function to plot the raw data
def plot_raw_data(paddock):
    paddock.plot(x='data', y='NDVI', figsize=(15, 6))
    plt.show()

# Function to resample data to weekly frequency and interpolate missing values
def resample_data(paddock):
    paddock_mean_date = paddock.groupby('data')['NDVI'].mean().reset_index()  # Mean by date
    paddock_mean_date.set_index('data', inplace=True)  # Set date as index
    paddock_w = paddock_mean_date.resample('7D').interpolate().reset_index()  # Weekly resampling
    return paddock_w

# Function to prepare data for the model
def prepare_model_data(paddock_w):
    paddock_model = paddock_w[['data', 'NDVI']].copy()
    paddock_model['unique_id'] = '1'
    paddock_model = paddock_model.rename(columns={'data': 'ds', 'NDVI': 'y'})
    return paddock_model

# Function to split data into training and test sets
def train_test_split(paddock_model):
    split = int(len(paddock_model) * 0.85)  # Split into 85% for training and 15% for testing
    paddock_model_train = paddock_model[:split]
    paddock_model_test = paddock_model[split:]
    return paddock_model_train, paddock_model_test

# Function to fit the model to the training data
def fit_model(paddock_model_train):
    model = sf(models=[aa(season_length=52)], freq='W')
    modelo_auto = model.fit(paddock_model_train)
    return modelo_auto

# Function to visualize residuals of the fitted model
def visualize_residuals(residual):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
    residual.plot(ax=axs[0, 0])
    axs[0, 0].set_title("Residuals")
    sns.distplot(residual, ax=axs[0, 1])
    axs[0, 1].set_title("Density plot - Residual")
    stats.probplot(residual["residual Model"], dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title('Plot Q-Q')
    plot_acf(residual, lags=35, ax=axs[1, 1], color="fuchsia")
    axs[1, 1].set_title("Autocorrelation")
    plt.show()

# Function to make forecasts using the fitted model
def forecast(modelo_auto):
    estimado = modelo_auto.predict(h=52, level=[95])
    return estimado

# Function to plot forecasts and observed data
def plot_forecast(paddock_model_train, paddock_model_test_filter, estimado):
    plt.figure(figsize=(10, 5))
    plt.plot(paddock_model_train['ds'], paddock_model_train['y'], color='blue', label='NDVI treino')
    plt.plot(estimado['ds'], estimado['AutoARIMA'], color='red', label='NDVI Estimado')
    plt.plot(paddock_model_test_filter['ds'], paddock_model_test_filter['y'], color='green', label='NDVI Observado')
    plt.xlabel('Data')
    plt.ylabel('NDVI')
    plt.legend()
    plt.show()

# Function to plot smoothed series
def plot_smoothed_series(paddock_model_train, paddock_model_test_filter, estimado, window_size=12):
    paddock_model_train['media_movel'] = paddock_model_train['y'].rolling(window=window_size).mean()
    estimado['media_movel'] = estimado['AutoARIMA'].rolling(window=window_size).mean()
    paddock_model_test_filter['media_movel'] = paddock_model_test_filter['y'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(paddock_model_train['ds'], paddock_model_train['media_movel'], color='blue', label='NDVI (training)')
    plt.plot(estimado['ds'], estimado['media_movel'], color='red', linestyle='--', label='NDVI (estimated)')
    plt.plot(paddock_model_test_filter['ds'], paddock_model_test_filter['media_movel'], color='green', label='NDVI (observed)')
    plt.xlabel('Data')
    plt.ylabel('NDVI')
    plt.legend()
    plt.show()

# Function to calculate Root Mean Squared Error (RMSE)
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

if __name__ == "__main__":
    # File path of the data
    file_path = '/Users/df_1.txt'

    # Reading the data
    paddock = read_data(file_path)

    # Preprocessing the data
    paddock = preprocess_data(paddock)

    # Plotting the raw data
    plot_raw_data(paddock)

    # Resampling the data
    paddock_w = resample_data(paddock)

    # Preparing the data for the model
    paddock_model = prepare_model_data(paddock_w)

    # Splitting the data into training and test sets
    paddock_model_train, paddock_model_test = train_test_split(paddock_model)

    # Fitting the model to the training data
    modelo_auto = fit_model(paddock_model_train)

    # Visualizing residuals of the fitted model
    residual = pd.DataFrame(modelo_auto.fitted_[0, 0].model_.get("residuals"), columns=["residual Model"])
    visualize_residuals(residual)

    # Making forecasts using the fitted model
    estimado = forecast(modelo_auto)

    # Plotting forecasts and observed data
    plot_forecast(paddock_model_train, paddock_model_test, estimado)

    # Plotting smoothed series
    plot_smoothed_series(paddock_model_train, paddock_model_test, estimado)

    # Calculating RMSE and MAE
    residual_values = residual["residual Model"].values
    rmse = calculate_rmse(np.zeros_like(residual_values), residual_values)
    mae = calculate_mae(np.zeros_like(residual_values), residual_values)

    print("RMSE:", rmse)
    print("MAE:", mae)
