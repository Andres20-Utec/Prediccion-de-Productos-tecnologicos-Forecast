import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .utils import *
from .data_processing import *
from .ml_functions import *
import os
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.getcwd()

class ForecasterPipeline:
    def __init__(self, df: pd.DataFrame, helper) -> None:
        """
        Inicializa la clase ForecasterPipeline.
        
        Parámetros:
            df (pd.DataFrame): DataFrame con las series temporales a predecir.
            helper: Objeto con métodos y atributos auxiliares, como configuración.
        """
        self.df = df
        self.helper = helper
        self.test_size = helper.semanas_a_predecir
        self.train = None
        self.test = None
        self.train_exog = None
        self.maxima_fecha = df['fecha'].max()

    def _objective_catboost(self, trial: optuna.Trial) -> float:
        """
        Objetivo para la optimización de CatBoostRegressor a través de Optuna.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e2),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 1e1),
            'loss_function': trial.suggest_categorical('loss_function', ['MAE', 'RMSE']),
            'posterior_sampling': trial.suggest_categorical('posterior_sampling', [True, False]),
        }

        model = CatBoostRegressor(**params, verbose=0)
        preds = train_and_predict(self.train, self.train_exog, self.test_size, model)

        y_true = self.test['y'].values
        y_pred = np.abs(preds['CatBoostRegressor'].values.round()).astype(int)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Criterio de parada temprana
        if mae < 10 and mape < 0.15:
            trial.study.stop()

        return mae

    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        """
        Objetivo para la optimización de XGBRegressor a través de Optuna.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'random_state': self.helper.seed,
        }

        model = XGBRegressor(**params, verbose=-1)
        preds = train_and_predict(self.train, self.train_exog, self.test_size, model)

        y_true = self.test['y'].values
        y_pred = np.abs(preds['XGBRegressor'].values.round()).astype(int)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Criterio de parada temprana
        if mae < 10 and mape < 0.15:
            trial.study.stop()

        return mae

    def _objective_lgbm(self, trial: optuna.Trial) -> float:
        """
        Objetivo para la optimización de LGBMRegressor a través de Optuna.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'random_state': self.helper.seed,
            'objective': 'regression_l2'
        }
        model = LGBMRegressor(**params, verbose=-1)
        preds = train_and_predict(self.train, self.train_exog, self.test_size, model)

        y_true = self.test['y'].values
        # Se usa el valor absoluto para evitar predicciones negativas
        y_pred = np.abs(preds['LGBMRegressor'].values.round()).astype(int)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Criterio de parada temprana
        if mae < 10 and mape < 0.15:
            trial.study.stop()

        return mae

    def _optimize_params(self, model_name: str, n_trials: int = 100) -> dict:
        """
        Corre la optimización de hiperparámetros para un modelo dado.

        Parámetros:
            model_name (str): Nombre del modelo ('LGBMRegressor', 'XGBRegressor', 'CatBoostRegressor').
            n_trials (int): Número de iteraciones de optimización.

        Retorna:
            dict: Diccionario con los mejores hiperparámetros encontrados.
        """
        objective_map = {
            'LGBMRegressor': self._objective_lgbm,
            'XGBRegressor': self._objective_xgboost,
            'CatBoostRegressor': self._objective_catboost
        }

        if model_name not in objective_map:
            raise ValueError("Modelo no soportado para optimización.")

        study = optuna.create_study(
            direction='minimize', 
            sampler=optuna.samplers.TPESampler(seed=self.helper.seed)
        )
        study.optimize(objective_map[model_name], n_trials=n_trials)

        return study.best_trial.params

    def _train_and_evaluate_models(self, fam: str, group: str, exogenous_vars: list, n_trials: int, 
                                   df_predicciones_lgbm: pd.DataFrame, 
                                   df_predicciones_xgb: pd.DataFrame, 
                                   df_predicciones_catboost: pd.DataFrame,
                                   df_resultados: pd.DataFrame) -> tuple:
        """
        Entrena y evalúa LGBM, XGB, y CatBoost para una familia dada. Retorna las 
        predicciones actualizadas, el df_resultados con las métricas, y los modelos ganadores.

        Parámetros:
            fam (str): Nombre de la familia.
            group (str): Grupo al que pertenece la familia.
            exogenous_vars (list): Lista de variables exógenas a utilizar.
            n_trials (int): Número de iteraciones de búsqueda de hiperparámetros.
            df_predicciones_lgbm (pd.DataFrame): DataFrame con predicciones LGBM.
            df_predicciones_xgb (pd.DataFrame): DataFrame con predicciones XGB.
            df_predicciones_catboost (pd.DataFrame): DataFrame con predicciones CatBoost.
            df_resultados (pd.DataFrame): DataFrame con resultados de métricas.

        Retorna:
            tuple: (df_predicciones_lgbm, df_predicciones_xgb, df_predicciones_catboost, df_resultados, best_model_name, best_params)
        """
        y_true = self.test['y'].values
        dates = self.test['ds'].values

        # Optimización de hiperparámetros para cada modelo
        best_params_lgbm = self._optimize_params('LGBMRegressor', n_trials)
        best_params_xgb = self._optimize_params('XGBRegressor', n_trials)
        best_params_cat = self._optimize_params('CatBoostRegressor', n_trials)

        model_lgb = LGBMRegressor(**best_params_lgbm, verbose=-1)
        model_xgb = XGBRegressor(**best_params_xgb, verbose=-1)
        model_cat = CatBoostRegressor(**best_params_cat, verbose=0)

        # Entrenar los tres modelos
        mlf = create_mlf([model_lgb, model_xgb, model_cat])
        mlf.fit(
            self.train,
            id_col='unique_id', 
            time_col='ds', 
            target_col='y', 
            static_features=[]
        )

        future_df = prepare_future_exog(mlf, self.train_exog, self.test_size)
        preds = mlf.predict(h=self.test_size, X_df=future_df)

        # Métricas
        mae_lgb = mean_absolute_error(y_true, preds['LGBMRegressor'])
        mape_lgb = mean_absolute_percentage_error(y_true, preds['LGBMRegressor'])

        mae_xgb = mean_absolute_error(y_true, preds['XGBRegressor'])
        mape_xgb = mean_absolute_percentage_error(y_true, preds['XGBRegressor'])

        mae_cat = mean_absolute_error(y_true, preds['CatBoostRegressor'])
        mape_cat = mean_absolute_percentage_error(y_true, preds['CatBoostRegressor'])

        # Resultados por modelo (DataFrames)
        df_predicciones_lgbm = self._append_predictions(
            df_predicciones_lgbm, 'LGBMRegressor', group, fam, y_true, dates, preds
        )
        df_predicciones_xgb = self._append_predictions(
            df_predicciones_xgb, 'XGBRegressor', group, fam, y_true, dates, preds
        )
        df_predicciones_catboost = self._append_predictions(
            df_predicciones_catboost, 'CatBoostRegressor', group, fam, y_true, dates, preds
        )

        # Agregar métricas a df_resultados
        modelos = ['LGBMRegressor', 'XGBRegressor', 'CatBoostRegressor']
        maes = [mae_lgb, mae_xgb, mae_cat]
        mapes = [mape_lgb, mape_xgb, mape_cat]

        for i, modelo in enumerate(modelos):
            df_resultados = pd.concat([
                df_resultados,
                pd.DataFrame([{
                    'Model': modelo,
                    'MAE': maes[i],
                    'MAPE': mapes[i],
                    'family': fam,
                    'group': group
                }])
            ], ignore_index=True)

        # Seleccionar mejor modelo
        best_model_name, best_result_per_month = get_better_models(
            [df_predicciones_lgbm, df_predicciones_xgb, df_predicciones_catboost], fam
        )

        # Si no hay resultado válido
        if best_result_per_month == float('inf'):
            return (df_predicciones_lgbm, df_predicciones_xgb, df_predicciones_catboost, df_resultados, None, None)

        # Determinar los mejores parámetros según el mejor modelo
        if best_model_name == 'LGBMRegressor':
            best_params = best_params_lgbm
        elif best_model_name == 'XGBRegressor':
            best_params = best_params_xgb
        else:
            best_params = best_params_cat

        return (df_predicciones_lgbm, df_predicciones_xgb, df_predicciones_catboost, df_resultados, best_model_name, best_params)

    def _append_predictions(self, df_preds: pd.DataFrame, model_name: str, group: str, fam: str, 
                            y_true: np.ndarray, dates: np.ndarray, preds: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega las predicciones de un modelo al DataFrame correspondiente.

        Parámetros:
            df_preds (pd.DataFrame): DataFrame donde se agregan las predicciones.
            model_name (str): Nombre del modelo.
            group (str): Nombre del grupo.
            fam (str): Nombre de la familia.
            y_true (np.ndarray): Valores verdaderos.
            dates (np.ndarray): Fechas del set de test.
            preds (pd.DataFrame): DataFrame con las predicciones devueltas por MLForecast.

        Retorna:
            pd.DataFrame: DataFrame con las predicciones agregadas.
        """
        y_pred = np.abs(preds[model_name].values.round()).astype(int)
        #error_percent = np.abs((y_true - y_pred) / y_true) * 100
        error_percent = np.where(
            y_true == 0,
            np.abs(y_pred) * 100,
            ((y_pred - y_true) / y_true) * 100
        )
        result_pred_model = pd.DataFrame({
            'date': dates,
            'y_real': y_true,
            'y_pred': y_pred,
            'error_percent': error_percent.round(2),
            'model_name': model_name,
            'group_name': group,
            'family_name': fam,
            'type': ['test' for _ in range(len(y_true))]
        })
        return pd.concat([df_preds, result_pred_model], ignore_index=True)

    def _prepare_data_for_family(self, cf_group: pd.DataFrame, nombre_grupo: str, fam: str) -> bool:
        """
        Prepara la data para una familia específica. Realiza comprobaciones, limpieza,
        interpolación y crea variables exógenas. Asigna self.train, self.test y self.train_exog.

        Parámetros:
            cf_group (pd.DataFrame): DataFrame filtrado por grupo.
            nombre_grupo (str): Nombre de la columna que define el grupo.
            fam (str): Nombre de la familia a procesar.

        Retorna:
            bool: True si la familia está lista para ser procesada, False en caso contrario.
        """
        data = cf_group[cf_group[nombre_grupo] == fam].reset_index(drop=True)
        data = data.rename(columns={'fecha': 'ds', 'cant_ventas': 'y', nombre_grupo: 'unique_id'})

        porcentaje_vacios = data['y'].isna().sum() / data.shape[0]
        perc_amount_zero = data[data['y'] == 0].shape[0] / data.shape[0]

        # Verificaciones de calidad de datos
        if (porcentaje_vacios > 0.1  or 
            data[data['ds'].isin([self.maxima_fecha])].empty or 
            perc_amount_zero >= 0.4):
            print("Familia con datos vacíos o nulos: ", fam)
            return False

        # Interpolación y relleno
        data = procesar_columna_y(data, 'y')

        # Eliminar filas con ceros iniciales
        data = eliminar_filas_ceros_iniciales(data, 'y')

        # Nuevas features
        data = agregar_features_peak(data, 'y')

        # Codificación de campaña
        data = reemplazar_valores(data, 'nombre_campana')

        data['prom_dolar'] = data['prom_dolar'].fillna(method='ffill')
        # Variables exógenas
        exogenous_vars = ['prom_dolar', 'nombre_campana', 'seasonal_peak', 'derivative_peak', 'is_peak']
        self.train_exog = data[exogenous_vars + ['unique_id', 'ds']]

        self.train = data.iloc[:-self.test_size]
        self.test = data.iloc[-self.test_size:]

        self.full_data = data  # Guardamos la data completa para entrenamiento final si se requiere
        self.exogenous_vars = exogenous_vars
        return True

    def _train_final_model(self, best_model_name: str, best_params: dict, test_future: pd.DataFrame, familia) -> pd.DataFrame:
        """
        Entrena el mejor modelo seleccionado con todos los datos disponibles y realiza predicciones futuras.

        Parámetros:
            best_model_name (str): Nombre del mejor modelo.
            best_params (dict): Hiperparámetros del mejor modelo.
            test_future (pd.DataFrame): DataFrame con información futura para crear variables exógenas.

        Retorna:
            pd.DataFrame: DataFrame con las predicciones a futuro.
        """
        if best_model_name == 'LGBMRegressor':
            model_selected = LGBMRegressor(**best_params, verbose=-1)
        elif best_model_name == 'XGBRegressor':
            model_selected = XGBRegressor(**best_params, verbose=-1)
        else:
            model_selected = CatBoostRegressor(**best_params, verbose=0)
        
        mlf_real = create_mlf([model_selected])

        mlf_real.fit(
            self.full_data,
            id_col='unique_id', 
            time_col='ds', 
            target_col='y', 
            static_features=[]
        )
        # Guardando el modelo
        guardar_modelo('src/pkl_modelos', model_selected, best_model_name, familia)

        # Generar PKL y enviarlo al S3
        sintetic_future = get_future_exog(test_future, self.exogenous_vars, self.test_size) # 12
        synthetic_base = mlf_real.make_future_dataframe(h=self.test_size) # 4 
        synthetic_future_complete = pd.merge(synthetic_base, sintetic_future, on=['unique_id', 'ds'], how='left') # 4
        pred_future = mlf_real.predict(h=self.test_size, X_df=synthetic_future_complete)
        return pred_future

    def run_pipeline(self) -> None:
        """
        Ejecuta el pipeline de forecasting para cada grupo configurado.
        Guarda resultados en CSV.
        """
        for group, n_trials in self.helper.grupo_trials.items():

            nombre_grupo = self.helper.obtener_nombre_grupo(group)

            # Obtener y preparar el DataFrame filtrado
            cf_group = obtener_tipo_informacion_agrupada(self.df, group, self.helper).copy()
            cf_group = completar_fechas_combinaciones(cf_group, col_fecha='fecha', col_familia=nombre_grupo)
            cf_group['fecha'] = pd.to_datetime(cf_group['fecha'], format='%Y-%m-%d')

            familias = cf_group[nombre_grupo].unique()

            # DataFrames para almacenar predicciones de cada modelo y futuras
            df_predicciones_catboost = obtener_dataframe_resultados()
            df_predicciones_lgbm = obtener_dataframe_resultados()
            df_predicciones_xgb = obtener_dataframe_resultados()
            df_predicciones_futuras = obtener_dataframe_resultados()

            df_resultados = pd.DataFrame(columns=['Model', 'MAE', 'MAPE','family', 'group'])

            for familia in familias:
                if not self._prepare_data_for_family(cf_group, nombre_grupo, familia):
                    continue
                print("Familia en proceso: ", familia)
                # Preparamos test_future para predicciones futuras
                # Este valor tiene que ser más grande que el test_size
                test_future = self.full_data.iloc[-12:]

                self.full_data.drop(columns=['nombre_campana', 'prom_dolar'], inplace=True)

                # Entrenar y evaluar modelos
                (df_predicciones_lgbm, 
                 df_predicciones_xgb, 
                 df_predicciones_catboost, 
                 df_resultados, 
                 best_model_name, 
                 best_params) = self._train_and_evaluate_models(
                     familia, group, self.exogenous_vars, n_trials, 
                     df_predicciones_lgbm, df_predicciones_xgb, 
                     df_predicciones_catboost, df_resultados
                 )

                # Si no se encontró un mejor modelo, continuar con la siguiente familia
                if best_model_name is None or best_params is None:
                    continue

                # Entrenar mejor modelo con todos los datos y predecir futuro
                pred_future = self._train_final_model(best_model_name, best_params, test_future, familia)

                # Crear DataFrame de predicciones futuras
                result_pred_future = pd.DataFrame({
                    'fecha': pred_future.ds,
                    'valor_real': [None for _ in range(self.test_size)],
                    'prediccion': np.abs(pred_future[best_model_name].values.round()).astype(int),
                    'porcentaje_error': [None for _ in range(self.test_size)],
                    'nombre_modelo': best_model_name,
                    'tipo_agrupacion': nombre_grupo,
                    'nombre_familia': familia,
                    'tipo': ['future' for _ in range(self.test_size)]
                })

                df_predicciones_futuras = pd.concat([df_predicciones_futuras, result_pred_future], ignore_index=True)

            # Guardar resultados de métricas
            # Tabla S3
            print(os.getcwd())
            df_resultados.to_csv(f"src/resultados_metricas_error/{group}_metricas_error.csv", index=False)
            # Guardar predicciones de cada modelo y futuras
            # Tabla Athena
            pd.concat(
                [df_predicciones_lgbm, df_predicciones_xgb, df_predicciones_catboost, df_predicciones_futuras],
                ignore_index=True
            ).to_csv(f"src/resultado_predicciones/{group}_predicciones.csv", index=False)
