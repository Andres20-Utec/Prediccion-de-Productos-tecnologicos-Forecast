import os
import pandas as pd
from .utils import Helper
from .data_processing import filtrar_columnas_con_insuficiente_informacion, filtros_iniciales
from .models import *
from .models import ForecasterPipeline
from ..data.preprocess import process_data_from_aws
from .athena import load_data # importar a tabla
from .boto import boto_client
from glob import glob
from dotenv import load_dotenv
load_dotenv()
class DirectoryManager:
    @staticmethod
    def create_directories(base_path, directories):
        for directory in directories:
            path = os.path.join(base_path, directory)
            os.makedirs(path, exist_ok=True)
            print(f"Directorio creado o existente: {path}")
        print("\nDirectorios creados con éxito.\n")


def run_pipeline():
    # Definición de rutas y archivos
    YAML_VARIABLES = "variables.yaml"
    YAML_MODELS = "modelos.yaml"
    DATA_FILE = "dataTable.csv"  # Ajustar la ruta según la estructura propuesta
    BASE_PATH = os.getcwd()

    # Crear directorios necesarios
    directories = [
        "src",  # Directorio base
        "src/resultado_predicciones",  # Resultados de las predicciones
        "src/resultados_metricas_error",  # Métricas de error por modelo
        "src/pkl_modelos",  # Modelos serializados
    ]

    # Intentar crear los directorios
    try:
        DirectoryManager.create_directories(BASE_PATH, directories)
    except Exception as e:
        print(f"Error al crear directorios: {e}")
        return {"status": "error", "message": f"Error al crear directorios: {e}"}

    try:
        # Cargar configuración desde los archivos YAML
        helper = Helper(
            variables_yaml_path=os.path.join("app/paquete", YAML_VARIABLES),
            modelos_yaml_path=os.path.join("app/paquete", YAML_MODELS)
        )
    except FileNotFoundError as e:
        print(f"Error al cargar archivos YAML: {e}")
        return {"status": "error", "message": f"Archivo YAML no encontrado: {e}"}
    except Exception as e:
        print(f"Error inesperado al cargar YAML: {e}")
        return {"status": "error", "message": f"Error al cargar YAML: {e}"}

    try:
        # Cargar y procesar datos
        print("Cargando datos...")
        df = pd.read_csv(os.path.join("app/src", DATA_FILE))
        #df = process_data_from_aws()

        # Filtrar datos
        df = filtrar_columnas_con_insuficiente_informacion(df)
        df = filtros_iniciales(df, helper)

        print("Datos cargados y procesados con éxito.")
        print(f"Tamaño del dataframe: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error al cargar el archivo de datos: {e}")
        return {"status": "error", "message": f"Archivo de datos no encontrado: {e}"}
    except Exception as e:
        print(f"Error inesperado al cargar y procesar los datos: {e}")
        return {"status": "error", "message": f"Error al cargar o procesar datos: {e}"}

    try:
        # Ejecutar el pipeline de forecasting
        print("Ejecutando el pipeline de forecasting...")
        forecast = ForecasterPipeline(df, helper)
        forecast.run_pipeline()
        print("Ejecución completada con éxito.")
    except Exception as e:
        print(f"Error al ejecutar el pipeline de forecasting: {e}")
        return {"status": "error", "message": f"Error al ejecutar el pipeline de forecasting: {e}"}

    try:
        # Subir resultados a Athena
        path = "src/resultado_predicciones"
        table_name = "forecasting_predicciones"
        folder = "forecasting"
        cols = ['fecha', 'valor_real', 'prediccion', 'porcentaje_error', 'nombre_modelo', 'tipo_agrupacion', 'nombre_familia', 'tipo']
        
        # Concatenar archivos de predicciones
        df_resultados = pd.concat([pd.read_csv(file) for file in glob(f"src/resultado_predicciones/*.csv")], ignore_index=True)
        load_data(df_resultados, table_name, folder, cols)
    except Exception as e:
        print(f"Error al cargar los resultados a Athena: {e}")
        return {"status": "error", "message": f"Error al cargar los resultados a Athena: {e}"}

    try:
        # Subir modelos serializados a S3
        bucket_name = os.getenv('AWS_S3_BUCKET')  # Nombre del bucket S3
        folder = generate_folder_name()
        table_name = "forecasting_modelos_pkl"
        path_pkl = "src/pkl_modelos"

        for file in glob(f"{path_pkl}/*.pkl"):
            nombre_file = file.split("/")[-1]
            boto_client.upload_file_to_s3(file, f"{bucket_name}/{folder}/{nombre_file}")
    except Exception as e:
        print(f"Error al subir los modelos serializados a S3: {e}")
        return {"status": "error", "message": f"Error al subir modelos a S3: {e}"}

    try:
        # Subir métricas de error del modelo a S3
        path_error_modelo = "src/resultados_metricas_error"
        table_name = "forecasting_error_modelo"
        for file in glob(f"{path_error_modelo}/*.csv"):
            nombre_file = file.split("/")[-1]
            boto_client.upload_file_to_s3(file, f"{bucket_name}/{folder}/{nombre_file}")
    except Exception as e:
        print(f"Error al subir métricas de error a S3: {e}")
        return {"status": "error", "message": f"Error al subir métricas de error a S3: {e}"}

    return {"status": "success", "message": "Pipeline ejecutado con éxito."}


