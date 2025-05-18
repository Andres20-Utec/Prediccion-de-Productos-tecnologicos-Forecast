from scipy.signal import find_peaks
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from mlforecast.lag_transforms import RollingMean, RollingStd, SeasonalRollingMean, SeasonalRollingMax, ExponentiallyWeightedMean
from mlforecast.target_transforms import Differences, LocalStandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from mlforecast import MLForecast

# ----------------------------
# Funciones de detección de picos
# ----------------------------

def seasonal_peaks(series, season_length=12, threshold=1.5):
    """
    Detecta picos estacionales en una serie temporal.
    
    Args:
        series (pd.Series): Serie temporal para analizar.
        season_length (int): Longitud del ciclo estacional (por defecto, 12 periodos).
        threshold (float): Multiplicador de desviación estándar para identificar picos.
    
    Returns:
        pd.Series: Serie binaria (1 para picos, 0 en caso contrario).
    """
    # Calcula la media y desviación estándar estacional
    seasonal_mean = series.shift(season_length).rolling(window=season_length, min_periods=1).mean()
    seasonal_std = series.shift(season_length).rolling(window=season_length, min_periods=1).std()
    # Marca un punto como pico si excede el umbral definido
    return ((series - seasonal_mean).abs() > threshold * seasonal_std).astype(int)


def derivative_peaks(series, threshold=1.5):
    """
    Identifica picos basados en cambios abruptos (derivadas) de la serie temporal.
    
    Args:
        series (pd.Series): Serie temporal para analizar.
        threshold (float): Multiplicador de desviación estándar para identificar picos.
    
    Returns:
        pd.Series: Serie binaria (1 para picos, 0 en caso contrario).
    """
    # Calcula la diferencia absoluta entre valores consecutivos
    diff = series.diff().abs()
    mean_diff = diff.mean()
    std_diff = diff.std()
    # Marca un punto como pico si la diferencia excede el umbral
    return (diff > mean_diff + threshold * std_diff).astype(int)


def detect_peaks(series, window_size=12, threshold=2.0):
    """
    Detecta picos en una serie temporal usando una ventana móvil.
    
    Args:
        series (pd.Series): Serie temporal para analizar.
        window_size (int): Tamaño de la ventana móvil.
        threshold (float): Multiplicador de desviación estándar para identificar picos.
    
    Returns:
        pd.Series: Serie binaria (1 para picos, 0 en caso contrario).
    """
    # Calcula media y desviación estándar móviles
    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, min_periods=1).std()
    # Marca un punto como pico si excede el umbral definido
    return ((series - rolling_mean).abs() > threshold * rolling_std).astype(int)


def detect_peaks_scipy(series, height=None, distance=12):
    """
    Detecta picos utilizando la biblioteca scipy.
    
    Args:
        series (pd.Series): Serie temporal para analizar.
        height (float): Altura mínima de los picos (opcional).
        distance (int): Distancia mínima entre picos consecutivos.
    
    Returns:
        np.array: Indicadores binarios para cada punto de la serie.
    """
    # Encuentra picos usando scipy
    peaks, _ = find_peaks(series, height=height, distance=distance)
    # Genera un indicador binario para los picos detectados
    peaks_indicator = np.zeros_like(series)
    peaks_indicator[peaks] = 1
    return peaks_indicator

def agregar_features_peak(data, column_y, season_length=12, threshold=1.5):
    """
    Agrega nuevas columnas al DataFrame basadas en la detección de picos:
    - 'is_peak': Detecta picos generales.
    - 'seasonal_peak': Detecta picos estacionales.
    - 'derivative_peak': Detecta picos basados en derivadas.

    :param data: DataFrame de entrada.
    :param column_y: Nombre de la columna que contiene los valores para detectar picos.
    :param season_length: Longitud de la temporada para los picos estacionales (default: 12).
    :param threshold: Umbral para los picos estacionales y de derivadas (default: 1.5).
    :return: DataFrame con las nuevas columnas añadidas.
    """
    # Nuevas features
    data['is_peak'] = detect_peaks(data[column_y])
    data['seasonal_peak'] = seasonal_peaks(data[column_y], season_length=season_length, threshold=threshold)
    data['derivative_peak'] = derivative_peaks(data[column_y], threshold=threshold)
    
    return data

# ----------------------------
# Funciones para variables exógenas futuras
# ----------------------------

def get_future_exog_by_column(data_exog, column, steps):
    """
    Predice valores futuros para una columna exógena específica.
    
    Args:
        data_exog (pd.DataFrame): Datos históricos de la variable exógena.
        column (str): Nombre de la columna a predecir.
        steps (int): Número de pasos futuros a predecir.
    
    Returns:
        pd.DataFrame: Predicciones futuras con fechas asociadas.
    """
    # Configura un modelo autorregresivo para predecir la columna
    forecaster = ForecasterAutoregMultiSeries(
        regressor=LinearRegression(n_jobs=-1),
        lags=8,  # Número de retardos usados para predecir
    )
    # Ajusta el modelo a los datos históricos
    forecaster.fit(series=data_exog[[column]])
    # Realiza las predicciones futuras
    predictions = forecaster.predict(steps=steps, levels=column)
    # Genera las fechas asociadas a las predicciones
    last_date = data_exog.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=steps, freq="W-MON")
    predictions.index = future_dates
    predictions.columns = [column]
    return predictions


def get_future_exog(df, col_var_exogenous, steps):
    """
    Genera valores futuros para todas las variables exógenas especificadas.
    
    Args:
        df (pd.DataFrame): DataFrame con las variables exógenas históricas.
        col_var_exogenous (list): Lista de nombres de columnas a predecir.
        steps (int): Número de pasos futuros a predecir.
    
    Returns:
        pd.DataFrame: Predicciones futuras para todas las variables exógenas.
    """
    # Preprocesa las variables exógenas
    data_exog = df[["ds", "unique_id"] + col_var_exogenous]
    data_exog["ds"] = pd.to_datetime(data_exog["ds"], format="%Y-%m-%d")
    data_exog = data_exog.set_index("ds").asfreq("W-MON", method="bfill")
    
    # Genera fechas futuras
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=steps, freq="W-MON")
    data_exog_future = pd.DataFrame(index=future_dates)
    
    # Predice para cada columna
    for column in col_var_exogenous:
        result_forecaster = get_future_exog_by_column(data_exog, column, steps)
        data_exog_future = pd.concat([data_exog_future, result_forecaster], axis=1)
    
    # Agrega identificadores únicos y fechas
    unique_id = df["unique_id"].iloc[0]
    data_exog_future["unique_id"] = unique_id
    data_exog_future["ds"] = data_exog_future.index
    data_exog_future = data_exog_future.reset_index(drop=True)
    return data_exog_future


# ----------------------------
# Funciones para configurar y entrenar modelos
# ----------------------------

def create_mlf(models):
    """
    Configura un pipeline de `MLForecast` para entrenar modelos de series temporales.
    
    Args:
        models (list): Lista de modelos de Machine Learning a usar en el pipeline.
    
    Returns:
        MLForecast: Objeto configurado para predicción de series temporales.
    """
    return MLForecast(
        models=models,
        freq='W-MON',
        lags=[1, 2, 3, 6, 12, 16],  # Definición de retardos
        lag_transforms={
            1: [RollingMean(window_size=3), RollingStd(window_size=3)],
            6: [SeasonalRollingMean(season_length=6, window_size=3)],
            12: [SeasonalRollingMax(season_length=12, window_size=3)],
            48: [ExponentiallyWeightedMean(alpha=0.3)],
        },
        date_features=['week', 'month', 'quarter', 'year'],  # Características temporales
        target_transforms=[Differences([1, 12]), LocalStandardScaler()],
        num_threads=6,  # Paralelismo
    )


def prepare_future_exog(mlf, train_exog, future_steps):
    """
    Prepara variables exógenas futuras para predecir con un pipeline de `MLForecast`.
    
    Args:
        mlf (MLForecast): Objeto de predicción configurado.
        train_exog (pd.DataFrame): Variables exógenas históricas.
        future_steps (int): Número de pasos futuros.
    
    Returns:
        pd.DataFrame: Variables exógenas futuras preparadas.
    """
    # Genera fechas futuras para predicción
    future_df = mlf.make_future_dataframe(h=future_steps)
    # Une las variables exógenas con las fechas futuras
    future_exog = train_exog[train_exog['ds'].isin(future_df['ds'])]
    future_df = pd.merge(future_df, future_exog, on=['unique_id', 'ds'], how='left')
    return future_df


def train_and_predict(train, train_exog, test_size, model):
    """
    Entrena un modelo y realiza predicciones para un horizonte definido.
    
    Args:
        train (pd.DataFrame): Datos de entrenamiento.
        train_exog (pd.DataFrame): Variables exógenas históricas.
        test_size (int): Tamaño del horizonte de predicción.
        model: Modelo de Machine Learning a entrenar.
    
    Returns:
        pd.DataFrame: Predicciones realizadas.
    """
    # Configura y entrena el pipeline
    mlf = create_mlf(models=[model])
    mlf.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
    # Prepara las variables exógenas futuras y realiza predicciones
    future_df = prepare_future_exog(mlf, train_exog, test_size)
    preds = mlf.predict(h=test_size, X_df=future_df)
    return preds

