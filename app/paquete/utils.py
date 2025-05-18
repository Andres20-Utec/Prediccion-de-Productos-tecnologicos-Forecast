import yaml
import pandas as pd
import numpy as np
import pickle
import datetime

class Helper:
    def __init__(self, variables_yaml_path, modelos_yaml_path):
        self.variables_yaml_path = variables_yaml_path
        self.modelos_yaml_path = modelos_yaml_path
        
        # Variables para almacenar la información cargada
        self.grupo_cols = None
        self.grupo_trials = None
        self.columnas_operacion = None
        self.columnas_exogenas_operacion = None
        self.variables_exogenas = None
        self.cadenas_sin_servicio = None
        self.modelos = None
        
        self.fecha_inicio = None
        self.seed = None
        self.semanas_a_predecir = None
        self.variable_objetivo = None

        # Cargar la información
        self.cargar_variables_yaml()
        self.cargar_modelos_yaml()

    def cargar_variables_yaml(self):
        """Carga la información del archivo variables.yaml"""
        with open(self.variables_yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            self.grupo_cols = data.get('grupo_cols', {})
            self.grupo_trials = data.get('grupo_trials', {})
            self.columnas_operacion = data.get('columnas_operacion', {})
            self.columnas_exogenas_operacion = data.get('columnas_exogenas_operacion', {})
            self.variables_exogenas = data.get('variables_exogenas', {})['variables']
            self.cadenas_sin_servicio = data.get('cadenas_sin_servicio', {})['cadenas']
            variables = data.get('config_variables', {})
            self.fecha_inicio = variables['fecha_inicio']
            self.seed = variables['seed']
            self.semanas_a_predecir = variables['semanas_a_predecir']
            self.variable_objetivo = variables['variable_objetivo']

    def cargar_modelos_yaml(self):
        """Carga la información del archivo modelos.yaml"""
        with open(self.modelos_yaml_path, 'r') as file:
            self.modelos = yaml.safe_load(file)

    def mostrar_informacion(self):
        """Muestra la información cargada"""
        print("Grupo de columnas:", self.grupo_cols)
        print("Grupo de trials:", self.grupo_trials)
        print("Columnas de operación:", self.columnas_operacion)
        print("Columnas exógenas de operación:", self.columnas_exogenas_operacion)
        print("Variables exógenas:", self.variables_exogenas)
        print("Cadenas sin servicio:", self.cadenas_sin_servicio)
        print("Modelos:", self.modelos)
        print("Fecha de inicio:", self.fecha_inicio)
        print("Seed:", self.seed)
        print("Semanas a predecir:", self.semanas_a_predecir)
        print("Variable objetivo:", self.variable_objetivo)
    
    def obtener_nombre_grupo(self, grupo):
        """Obtiene el nombre del grupo"""
        return "_".join(self.grupo_cols[grupo])

# No toma en cuenta la cantidad de meses, solo el primer mes
def get_better_models(models_df_results, family, n_group=4):
    # Fija el rango del primer mes (0 a 4 semanas)
    week_start = 0
    week_end = n_group  # n_group define el rango del primer mes (4 semanas)
    
    results_error_percent = []
    for results in models_df_results:  # Itera por los dataframes de predicciones
        model_name = results['model_name'].unique()[0]
        data = results[results['family_name'] == family]
        data_month = data.iloc[week_start:week_end]  # Datos del primer mes
        data_month['error_percent'] = np.abs(data_month['error_percent'])
        # Calcula el porcentaje de error promedio para el primer mes
        error_percent_month = data_month.error_percent.mean()
        results_error_percent.append((model_name, error_percent_month))
    
    # Selecciona el modelo con menor porcentaje de error para el primer mes
    best_model, best_percent = min(results_error_percent, key=lambda x: x[1])
    return best_model, best_percent


def obtener_dataframe_resultados():
    return pd.DataFrame(columns=['fecha', 'valor_real', 'prediccion', 'porcentaje_error', 'nombre_modelo', 'tipo_agrupacion', 'nombre_familia', 'tipo'])


def guardar_modelo(path, model, model_name, type):
    with open(f"{path}/{model_name}_{type}.pkl", 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo guardado en {path}/{model_name}_{type}.pkl")

def generate_folder_name(date=None):
    """
    Genera un nombre basado en el año, mes y semana de una fecha dada o la fecha actual.

    Args:
        date (datetime.date, optional): La fecha de referencia. Si no se proporciona, se usará la fecha actual.
    
    Returns:
        str: El nombre en el formato "Y<year>M<month>W<week_number>".
    """
    # Usa la fecha actual si no se proporciona una fecha
    if date is None:
        date = datetime.date.today()
    
    # Obtiene el año y el mes
    year = date.year
    month = date.month
    
    # Calcula el número de semana basado en el estándar ISO (lunes como primer día de la semana)
    week_number = date.isocalendar()[1]
    
    # Formatea el nombre
    name = f"Y{year}_M{month}_WN{week_number}"
    return name
