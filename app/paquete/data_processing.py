import pandas as pd
import numpy as np
import os


def filtros_iniciales(data, helper):
    data = data[data['fecha'] >= helper.fecha_inicio].reset_index(drop=True)
    data = data[~data['familia'].str.contains("OTROS")]
    cadenas_sin_servicio = helper.cadenas_sin_servicio
    data = data[~data['cadena'].isin(cadenas_sin_servicio)].reset_index(drop=True)
    data[helper.variable_objetivo] = data[helper.variable_objetivo].abs() # .fillna(0.0)
    return data


def completar_fechas_combinaciones(df, col_fecha ='fecha', col_familia = 'familia'):
    """
    Completa las combinaciones faltantes de fechas y familia en el DataFrame.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original que contiene las columnas de fecha y familia.
    col_fecha : str
        Nombre de la columna que contiene las fechas.
    col_familia : str
        Nombre de la columna que contiene las combinaciones (familia).

    Retorna:
    --------
    pd.DataFrame
        DataFrame con todas las combinaciones posibles de fecha y familia, llenando con NaN donde falten datos.
    """
    # Obtener fechas y combinaciones únicas
    fechas_unicas = df[col_fecha].unique()
    combinaciones_unicas = df[col_familia].unique()

    # Crear un DataFrame con todas las combinaciones posibles de fecha y familia
    fechas_combinaciones = pd.MultiIndex.from_product(
        [fechas_unicas, combinaciones_unicas], names=[col_fecha, col_familia]
    ).to_frame(index=False)

    # Hacer un merge con el DataFrame original para llenar las combinaciones faltantes
    df_completo = pd.merge(fechas_combinaciones, df, on=[col_fecha, col_familia], how='left')

    # Ordenar el DataFrame por familia y fecha
    df_completo = df_completo.sort_values(by=[col_familia, col_fecha]).reset_index(drop=True)

    return df_completo

# Revisar función
def filtrar_columnas_con_insuficiente_informacion(df, umbral=0.80):
    """
    Mantiene las columnas que tienen más del 80% de datos no nulos y no vacíos.

    Parámetros:
    df (pandas.DataFrame): El DataFrame de entrada.
    umbral (float): El umbral de porcentaje de valores no nulos y no vacíos (por defecto 0.80).

    Retorna:
    pandas.DataFrame: Un DataFrame con las columnas que cumplen con el umbral de información.
    """
    # Contar valores no nulos y no vacíos en cada columna
    conteo_valido = df.apply(lambda col: col.notna() & (col != '')).sum()
    
    # Calcular el total de filas en el DataFrame
    total_filas = len(df)
    
    # Calcular el porcentaje de valores válidos
    porcentaje_valido = conteo_valido / total_filas
    
    # Filtrar las columnas que cumplen con el umbral del 80%
    columnas_a_conservar = porcentaje_valido[porcentaje_valido > umbral].index
    
    # Retornar el DataFrame solo con las columnas seleccionadas
    return df[columnas_a_conservar]


def obtener_tipo_informacion_agrupada(df, tipo, helper):
    """
    Agrupa la información del DataFrame basado en columnas específicas.

    Args:
        data (pd.DataFrame): DataFrame de entrada que contiene los datos a agrupar.
        group_columns (list): Lista de columnas para realizar la agrupación. 
                              Por ejemplo, ["CF", "CS"] donde:
                              - "CF": Agrupa por cadena_familia.
                              - "CS": Agrupa por cadena_subfamilia.
    
    Returns:
        dict: Diccionario donde las claves son los nombres de los grupos
              (por ejemplo, "CF", "CS") y los valores son DataFrames agrupados.
    """
    # Columnas de operación
    var_op = dict(helper.columnas_operacion).copy()
    # Agregar columnas exógenas de operación
    var_op.update(helper.columnas_exogenas_operacion)

    # Obtener el nombre del grupo dinámicamente
    grupo_col = helper.obtener_nombre_grupo(tipo)
    
    # Construir columna agrupada dinámicamente según tipo
    if tipo in helper.grupo_cols:
        cols_a_concatenar = helper.grupo_cols[tipo]
        df[grupo_col] = df[cols_a_concatenar].astype(str).agg('_'.join, axis=1)
    
    # Filtrar columnas relevantes
    columnas_relevantes = ['fecha', grupo_col, 'cant_ventas'] + helper.variables_exogenas
    df = df[columnas_relevantes]
    
    # Agrupar y devolver
    return (
        df.groupby(['fecha', grupo_col])
        .agg(var_op, min_count=-1)
        .reset_index()
        .sort_values(by=[grupo_col, 'fecha'])
        .reset_index(drop=True)
    )

def eliminar_filas_ceros_iniciales(df, columna_y):
    """
    Elimina las filas desde el inicio hasta encontrar un valor mayor que cero en la columna especificada.
    
    :param df: DataFrame ordenado por fecha.
    :param columna_y: Nombre de la columna donde se revisarán los valores.
    :return: DataFrame filtrado.
    """
    # Encuentra el índice de la primera fila donde el valor en columna_y es mayor a cero
    idx_inicio = df[df[columna_y] > 0].index.min()
    
    # Si no se encuentra ningún valor mayor a cero, devuelve un DataFrame vacío
    if pd.isna(idx_inicio):
        return pd.DataFrame(columns=df.columns)
    
    # Devuelve el DataFrame a partir del índice encontrado
    return df.loc[idx_inicio:].reset_index(drop=True)

def reemplazar_valores(df, columna):
    """
    Reemplaza los valores en la columna especificada:
    - 'Regular', NaN o Null -> 0
    - Cualquier otro valor -> 1
    
    :param df: DataFrame de entrada.
    :param columna: Nombre de la columna a procesar.
    :return: DataFrame con la columna modificada.
    """
    # Reemplazar valores según la condición
    df[columna] = df[columna].apply(lambda x: 0 if pd.isna(x) or x == 'Regular' else 1)
    return df

def procesar_columna_y(data, columna_y):
    """
    Procesa la columna especificada en el DataFrame aplicando las siguientes transformaciones:
    - Interpolación polinomial de orden 3.
    - Relleno de valores nulos con 0.
    - Aplicación de valor absoluto.
    - Redondeo a enteros.
    - Conversión a tipo entero.

    :param data: DataFrame de entrada.
    :param columna_y: Nombre de la columna a procesar.
    :return: DataFrame con la columna procesada.
    """
    data[columna_y] = (
        data[columna_y]
        .interpolate(method='polynomial', order=3)  # Interpolación polinomial
        .fillna(0)                                  # Llena valores nulos iniciales con 0
        .abs()                                      # Aplica valor absoluto
        .round(0)                                   # Redondea los valores
        .astype(int)                                # Convierte a tipo entero
    )
    return data