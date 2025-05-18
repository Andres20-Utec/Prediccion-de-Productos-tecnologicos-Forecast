import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def custom_agg(func):
        def agg_func(series):
            return func(series) if series.notna().any() else np.nan
        return agg_func

# ---------------------------- Seasonal Decompose ----------------------------

def family_seasonal_decompose(df):
    df['prom_dolar'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df['prom_precio'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df.drop(columns=['nombre_pdv', 'cadena', 'codigo_pdv', 'subfamilia'], inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Agrupar en base a categoria
    df_filter = df.groupby(['fecha', 'familia']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    # Completas fechas faltantes

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    # Obtener todas las fechas únicas y combinaciones únicas
    fechas_unicas = df_filter['fecha'].unique()
    combinaciones_unicas = df_filter['familia'].unique()

    # Crear un DataFrame con todas las combinaciones posibles de fecha y combinacion
    fechas_combinaciones = pd.MultiIndex.from_product(
        [fechas_unicas, combinaciones_unicas], names=['fecha', 'familia']
    ).to_frame(index=False)

    # Hacer un merge con el DataFrame original para llenar las combinaciones faltantes
    df_filter = pd.merge(fechas_combinaciones, df_filter, on=['fecha', 'familia'], how='left')

    # Ordenar por 'combinacion' y 'fecha'
    df_filter = df_filter.sort_values(by=['familia', 'fecha']).reset_index(drop=True)

    # Crear un DataFrame vacío para almacenar todos los resultados
    resultados_decompose = pd.DataFrame()

    # Obtener una lista de combinaciones únicas
    combinaciones_unicas = df_filter['familia'].unique()
    fechas_unicas = len(df_filter['fecha'].unique())

    # Iterar sobre cada combinación
    for combinacion in combinaciones_unicas:
        # Filtrar el DataFrame para cada combinación
        df_comb = df_filter[df_filter['familia'] == combinacion]
        
        print(combinacion, df_comb['cant_ventas'].isna().sum())

        if df_comb['cant_ventas'].isna().sum() / fechas_unicas <= 0.1:
            # Ordenar por fecha
            df_comb = df_comb.sort_values(by='fecha')

            # Interpolar los valores faltantes
            df_comb['cant_ventas'] = df_comb['cant_ventas'].interpolate(method='polynomial', order=3)

            # Rellenar cualquier valor nulo restante en los extremos después de la interpolación
            df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(method='bfill').fillna(method='ffill')

            # Verificar si hay valores nulos restantes
            if df_comb['cant_ventas'].isna().sum() > 30:
                print(f"Advertencia: La combinación {combinacion} aún tiene valores nulos. Se procederá a rellenarlos con ceros.")
                #df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(0)

            # Additive Decomposition
            result_add = seasonal_decompose(df_comb['cant_ventas'], model='additive', period=12)

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición aditiva
            df_add = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_add': result_add.observed,
                'trend_add': result_add.trend,
                'seasonal_add': result_add.seasonal,
                'residual_add': result_add.resid
            })

            # Multiplicative Decomposition
            try:
                result_mul = seasonal_decompose(df_comb['cant_ventas'], model='multiplicative', period=12)
            except ValueError as e:
                print(f"No se puede realizar la descomposición multiplicativa para la combinación {combinacion}: {e}")
                continue

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición multiplicativa
            df_mul = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_mul': result_mul.observed,
                'trend_mul': result_mul.trend,
                'seasonal_mul': result_mul.seasonal,
                'residual_mul': result_mul.resid
            })

            # Unir ambos DataFrames temporales (aditivo y multiplicativo)
            df_result = pd.merge(df_add, df_mul, on=['fecha', 'combinacion'])

            # Concatenar el resultado al DataFrame general
            resultados_decompose = pd.concat([resultados_decompose, df_result], ignore_index=True)

    resultados_decompose.to_csv('familia_seasonal_decompose.csv', index=False)

def subfamily_seasonal_decompose(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.drop(columns=['nombre_pdv', 'cadena', 'codigo_pdv', 'familia'], inplace=True)
    df.sort_values(by=['fecha', 'subfamilia'], inplace=True)

    # Aplica la función personalizada para cada tipo de agregación
    df_filter = df.groupby(['fecha', 'subfamilia']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    # Obtener todas las fechas únicas y combinaciones únicas
    fechas_unicas = df_filter['fecha'].unique()
    combinaciones_unicas = df_filter['subfamilia'].unique()

    # Crear un DataFrame con todas las combinaciones posibles de fecha y combinacion
    fechas_combinaciones = pd.MultiIndex.from_product(
        [fechas_unicas, combinaciones_unicas], names=['fecha', 'subfamilia']
    ).to_frame(index=False)

    # Hacer un merge con el DataFrame original para llenar las combinaciones faltantes
    df_filter = pd.merge(fechas_combinaciones, df_filter, on=['fecha', 'subfamilia'], how='left')

    # Ordenar por 'combinacion' y 'fecha'
    df_filter = df_filter.sort_values(by=['subfamilia', 'fecha']).reset_index(drop=True)
    
    # Crear un DataFrame vacío para almacenar todos los resultados
    resultados_decompose = pd.DataFrame()

    # Obtener una lista de combinaciones únicas
    combinaciones_unicas = df_filter['subfamilia'].unique()
    fechas_unicas = len(df_filter['fecha'].unique())

    # Iterar sobre cada combinación
    for combinacion in combinaciones_unicas:
        # Filtrar el DataFrame para cada combinación
        df_comb = df_filter[df_filter['subfamilia'] == combinacion]
        
        print(combinacion, df_comb['cant_ventas'].isna().sum())

        if df_comb['cant_ventas'].isna().sum() / fechas_unicas <= 0.1:
            # Ordenar por fecha
            df_comb = df_comb.sort_values(by='fecha')

            # Interpolar los valores faltantes
            df_comb['cant_ventas'] = df_comb['cant_ventas'].interpolate(method='polynomial', order=3)

            # Rellenar cualquier valor nulo restante en los extremos después de la interpolación
            df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(method='bfill').fillna(method='ffill')

            # Verificar si hay valores nulos restantes
            if df_comb['cant_ventas'].isna().sum() > 30:
                print(f"Advertencia: La combinación {combinacion} aún tiene valores nulos. Se procederá a rellenarlos con ceros.")
                #df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(0)

            # Additive Decomposition
            result_add = seasonal_decompose(df_comb['cant_ventas'], model='additive', period=12)

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición aditiva
            df_add = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_add': result_add.observed,
                'trend_add': result_add.trend,
                'seasonal_add': result_add.seasonal,
                'residual_add': result_add.resid
            })

            # Multiplicative Decomposition
            try:
                result_mul = seasonal_decompose(df_comb['cant_ventas'], model='multiplicative', period=12)
            except ValueError as e:
                print(f"No se puede realizar la descomposición multiplicativa para la combinación {combinacion}: {e}")
                continue

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición multiplicativa
            df_mul = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_mul': result_mul.observed,
                'trend_mul': result_mul.trend,
                'seasonal_mul': result_mul.seasonal,
                'residual_mul': result_mul.resid
            })

            # Unir ambos DataFrames temporales (aditivo y multiplicativo)
            df_result = pd.merge(df_add, df_mul, on=['fecha', 'combinacion'])

            # Concatenar el resultado al DataFrame general
            resultados_decompose = pd.concat([resultados_decompose, df_result], ignore_index=True)

    resultados_decompose.to_csv('subfamilia_seasonal_decompose.csv', index=False)

     

def cadena_family_seasonal_decompose(df):
    df['prom_dolar'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df['prom_precio'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.drop(columns=['nombre_pdv', 'codigo_pdv', 'subfamilia'], inplace=True)
    df.sort_values(by=['fecha', 'cadena', 'familia'], inplace=True)
    df['combinacion'] = df['cadena'] + '_' + df['familia']
    df.drop(columns=['cadena', 'familia'], inplace=True)

    # Aplica la función personalizada para cada tipo de agregación
    df_filter = df.groupby(['fecha', 'combinacion']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    # Obtener todas las fechas únicas y combinaciones únicas
    fechas_unicas = df_filter['fecha'].unique()
    combinaciones_unicas = df_filter['combinacion'].unique()

    # Crear un DataFrame vacío para almacenar todos los resultados
    resultados_decompose = pd.DataFrame()
    resultados_corr = pd.DataFrame()

    # Obtener una lista de combinaciones únicas
    combinaciones_unicas = df_filter['combinacion'].unique()
    fechas_unicas = len(df_filter['fecha'].unique())

    # Iterar sobre cada combinación
    for combinacion in combinaciones_unicas:
        # Filtrar el DataFrame para cada combinación
        df_comb = df_filter[df_filter['combinacion'] == combinacion]
        
        print(combinacion, df_comb['cant_ventas'].isna().sum())

        if df_comb['cant_ventas'].isna().sum() / fechas_unicas <= 0.1:
            # Ordenar por fecha
            df_comb = df_comb.sort_values(by='fecha')

            # Interpolar los valores faltantes
            df_comb['cant_ventas'] = df_comb['cant_ventas'].interpolate(method='polynomial', order=3)

            # Rellenar cualquier valor nulo restante en los extremos después de la interpolación
            df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(method='bfill').fillna(method='ffill')

            # Verificar si hay valores nulos restantes
            if df_comb['cant_ventas'].isna().sum() > 30:
                print(f"Advertencia: La combinación {combinacion} aún tiene valores nulos. Se procederá a rellenarlos con ceros.")
                #df_comb['cant_ventas'] = df_comb['cant_ventas'].fillna(0)

            # Additive Decomposition
            result_add = seasonal_decompose(df_comb['cant_ventas'], model='additive', period=12)

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición aditiva
            df_add = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_add': result_add.observed,
                'trend_add': result_add.trend,
                'seasonal_add': result_add.seasonal,
                'residual_add': result_add.resid
            })

            # Multiplicative Decomposition
            try:
                result_mul = seasonal_decompose(df_comb['cant_ventas'], model='multiplicative', period=12)
            except ValueError as e:
                print(f"No se puede realizar la descomposición multiplicativa para la combinación {combinacion}: {e}")
                continue

            # Crear un DataFrame temporal para almacenar los resultados de la descomposición multiplicativa
            df_mul = pd.DataFrame({
                'fecha': df_comb['fecha'],
                'combinacion': combinacion,
                'observed_mul': result_mul.observed,
                'trend_mul': result_mul.trend,
                'seasonal_mul': result_mul.seasonal,
                'residual_mul': result_mul.resid
            })

            # Unir ambos DataFrames temporales (aditivo y multiplicativo)
            df_result = pd.merge(df_add, df_mul, on=['fecha', 'combinacion'])

            # Concatenar el resultado al DataFrame general
            resultados_decompose = pd.concat([resultados_decompose, df_result], ignore_index=True)

    resultados_decompose.to_csv('CF_seasonal_decompose.csv', index=False)
     
def cadena_subfamily_seasonal_decompose(df):
    print("cadena_subfamily_seasonal_decompose")
    
     
# ---------------------------- Correlation ----------------------------
    
def family_correlation(df):
    df.drop(columns=['nombre_pdv', 'cadena', 'codigo_pdv', 'subfamilia'], inplace=True)

    # Agrupar en base a categoria
    df_filter = df.groupby(['fecha', 'familia']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    combinaciones_unicas = df_filter['familia'].unique()
    correlation_results = []

    for combinacion in combinaciones_unicas:
        df_comb = df_filter[(df_filter['familia'] == combinacion) & (df_filter['fecha'] >= '2019-01-01')]
        df_comb = df_comb.sort_values(by='fecha')
        df_corr = df_comb.drop(columns=['fecha', 'familia'])
        if 'cant_ventas' in df_corr.columns:
            corr_result = df_corr.corrwith(df_corr['cant_ventas'], axis=0, method='spearman')
            for variable, value in corr_result.items():
                correlation_results.append({
                    'combinacion': combinacion,
                    'variable_correlacion': variable,
                    'valor_correlacion': value
                })
    
    correlation_df = pd.DataFrame(correlation_results)

    correlation_df.to_csv('familia_correlation_results.csv', index=False)


def subfamily_correlation(df):
    df.drop(columns=['nombre_pdv', 'cadena', 'codigo_pdv', 'familia'], inplace=True)
    df.sort_values(by=['fecha', 'subfamilia'], inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Aplica la función personalizada para cada tipo de agregación
    df_filter = df.groupby(['fecha', 'subfamilia']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    # Obtener una lista de combinaciones únicas
    combinaciones_unicas = df_filter['subfamilia'].unique()
    fechas_unicas = len(df_filter['fecha'].unique())

    # Lista para almacenar todos los resultados
    correlation_results = []

    for combinacion in combinaciones_unicas:
        # Filtrar los datos por la combinación y el rango de fechas requerido
        df_comb = df_filter[(df_filter['subfamilia'] == combinacion) & (df_filter['fecha'] >= '2019-01-01')]

        # Ordenar por fecha
        df_comb = df_comb.sort_values(by='fecha')

        # Asegúrate de filtrar las columnas correctas antes de calcular la correlación
        df_corr = df_comb.drop(columns=['fecha', 'subfamilia'])

        # Verifica que la columna 'cant_ventas' esté presente
        if 'cant_ventas' in df_corr.columns:
            # Calcula la correlación de cada columna con la columna 'cant_ventas'
            corr_result = df_corr.corrwith(df_corr['cant_ventas'], axis=0, method='spearman')

            # Convierte la correlación a un DataFrame y añade la combinación
            for variable, value in corr_result.items():
                correlation_results.append({
                    'combinacion': combinacion,
                    'variable_correlacion': variable,
                    'valor_correlacion': value
                })

    # Convertir los resultados a un DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Guardar el DataFrame en un archivo CSV
    correlation_df.to_csv('subfamilia_correlation_results.csv', index=False)


def cadena_subfamily_correlation(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['prom_dolar'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df['prom_precio'] = pd.to_numeric(df['prom_dolar'], errors='coerce')
    df.drop(columns=['nombre_pdv', 'codigo_pdv', 'subfamilia'], inplace=True)
    df['combinacion'] = df['cadena'] + '_' + df['familia']
    df.drop(columns=['cadena', 'familia'], inplace=True)
    
    # Aplica la función personalizada para cada tipo de agregación
    df_filter = df.groupby(['fecha', 'combinacion']).agg({
        # Columnas que contienen "cant" -> se suman
        **{col: custom_agg(np.sum) for col in df.columns if 'cant' in col},
        
        # Columnas que contienen "prom" -> se promedian
        **{col: custom_agg(np.mean) for col in df.columns if 'prom' in col},

        # Columnas que contienen "nombre" -> se toma el primer valor
        **{col: custom_agg(lambda x: x.iloc[0]) for col in df.columns if 'nombre' in col}
    }).reset_index()

    label_encoder_campana = {'Regular': 1, 'Back To School': 2, 'Cyber Week': 3, 'Cyber Wow': 4, 'Cyber Days': 5, 'Black Week': 6, 'Fiestas Navideñas': 7}
    df_filter['nombre_campana'] = df_filter['nombre_campana'].map(label_encoder_campana)

    # Obtener todas las fechas únicas y combinaciones únicas
    fechas_unicas = df_filter['fecha'].unique()
    combinaciones_unicas = df_filter['combinacion'].unique()
    
    # Lista para almacenar todos los resultados
    correlation_results = []

    for combinacion in combinaciones_unicas:
        # Filtrar los datos por la combinación y el rango de fechas requerido
        df_comb = df_filter[(df_filter['combinacion'] == combinacion) & (df_filter['fecha'] >= '2019-07-08')]

        # Ordenar por fecha
        df_comb = df_comb.sort_values(by='fecha')

        # Asegúrate de filtrar las columnas correctas antes de calcular la correlación
        df_corr = df_comb.drop(columns=['fecha', 'combinacion'])

        # Verifica que la columna 'cant_ventas' esté presente
        if 'cant_ventas' in df_corr.columns:
            # Calcula la correlación de cada columna con la columna 'cant_ventas'
            corr_result = df_corr.corrwith(df_corr['cant_ventas'], axis=0, method='spearman')

            # Convierte la correlación a un DataFrame y añade la combinación
            for variable, value in corr_result.items():
                correlation_results.append({
                    'combinacion': combinacion,
                    'variable_correlacion': variable,
                    'valor_correlacion': value
                })

    # Convertir los resultados a un DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Guardar el DataFrame en un archivo CSV
    correlation_df.to_csv('CF_correlation_results.csv', index=False)