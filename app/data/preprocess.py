import logging.config
import time
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
from .load_data import *
from .helper import *
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------- Data Connection ----------------- #

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY') #accessKey
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY') #secretKey
SCHEMA_NAME = os.getenv('SCHEMA_NAME') #schema
S3_STAGING_DIR = os.getenv('S3_STAGING_DIR') #keyS3
AWS_REGION = os.getenv('AWS_REGION') #keyRegion

class AthenaClient:
    def __init__(self):
        access_key = AWS_ACCESS_KEY
        secret_key = AWS_SECRET_KEY
        s3_staging_dir = S3_STAGING_DIR
        schema_name = SCHEMA_NAME

        assert access_key is not None, "AWS_ACCESS_KEY_ID no está definido en las variables de entorno"
        assert secret_key is not None, "AWS_SECRET_ACCESS_KEY no está definido en las variables de entorno"
        assert s3_staging_dir is not None, "S3_STAGING_DIR no está definido en las variables de entorno"
        assert schema_name is not None, "ATHENA_SCHEMA_NAME no está definido en las variables de entorno"

        conn_str = (
            "awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@"
            "athena.{region_name}.amazonaws.com:443/"
            "{schema_name}?s3_staging_dir={s3_staging_dir}&work_group=primary"
        )

        # SQLAlchemy connection con PyAthena
        self.engine = create_engine(
            conn_str.format(
                aws_access_key_id=quote_plus(AWS_ACCESS_KEY),
                aws_secret_access_key=quote_plus(AWS_SECRET_KEY),
                region_name=AWS_REGION,
                schema_name=SCHEMA_NAME,
                s3_staging_dir=quote_plus(S3_STAGING_DIR),
            )
        )

engine = AthenaClient().engine

lenovo_competencia_lista = ['HP', 'ACER', 'ASUS', 'SAMSUNG', 'TCL', 'XIAOMI', 'HONOR']

# ----------------- Lenovo Data Preprocessing ----------------- #

def preprocessVentas(df_ventas):
    df_ventas = df_ventas.sort_values(by='fecha')
    df_ventas = df_ventas.reset_index(drop=True)

    df_ventas['subfamilia'] = np.where((df_ventas['subfamilia'] == '-') & (df_ventas['familia'] == 'NOTEBOOK'), 'Portafolio NB', np.where((df_ventas['subfamilia'] == '-') & (df_ventas['familia'] == 'TABLET'), 'Portafolio TB', np.where((df_ventas['subfamilia'] == '-') & ((df_ventas['familia'] == 'DT - TOWER')|(df_ventas['familia'] == 'ACCESORIO')), 'OTROS', df_ventas['subfamilia'])))
    df_ventas['subfamilia'] = np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'NOTEBOOK'), 'Portafolio NB', np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'TABLET'), 'Portafolio TB', np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'DT - AIO'),'AIO',np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'ACCESORIO'),'OTROS',df_ventas['subfamilia']))))

    familia_mapping_dict = {
        'TABLET': 'TABLET',
        'ACCESORIO': 'OTROS',
        'NOTEBOOK': 'NOTEBOOK',
        'DT - AIO': 'DT - AIO',
        'GAMING CONSOLE': 'OTROS',
        'WORKSTATION': 'OTROS',
        'DT - TOWER': 'OTROS',
        'Liquidación': 'OTROS',
        'AIO': 'DT - AIO'
    }

    subfamilia_mapping_dict = {
        'Portafolio TB': 'PORTAFOLIO TB',
        'YOGA TABLET': 'YOGA TABLET',
        'PHABLET': 'PHABLET',
        'AIO': 'AIO',
        'Portafolio NB': 'PORTAFOLIO NB',
        'Gaming': 'GAMING',
        'THINKBOOK': 'THINKBOOK',
        'THINKPAD': 'THINKPAD',
        '2 en 1': '2_en_1',
        '2 EN 1': '2_en_1',
        '-': 'OTROS',
        '0': 'OTROS',
        'THINKSTATION': 'OTROS',
        'Liquidación': 'OTROS',
        'ACCESORIO': 'OTROS',
        'OTROS': 'OTROS'
    }

    df_ventas['familia'] = df_ventas['familia'].map(familia_mapping_dict)
    df_ventas['subfamilia'] = df_ventas['subfamilia'].map(subfamilia_mapping_dict)
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
    df_ventas['cant_ventas'] = pd.to_numeric(df_ventas['cant_ventas'])
    df_ventas['prom_precio_b2b'] = pd.to_numeric(df_ventas['prom_precio_b2b'])
    df_ventas.sort_values(by=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia'], inplace=True)

    print("Lenovo sales preprocesed")
    return df_ventas

def preprocessPrecio(df_precio):
    df_precio = df_precio.sort_values(by=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia'])
    df_precio = df_precio.reset_index(drop=True)
    df_precio['fecha'] = pd.to_datetime(df_precio['fecha'])
    df_precio['prom_precio'] = pd.to_numeric(df_precio['prom_precio'])

    familia_mapping_dict = {
        'AIO': 'DT - AIO',
        'NOTEBOOK': 'NOTEBOOK',
        'TABLET': 'TABLET',
        'CUR_PR14': 'TABLET',
        'EFE_PR05': 'DT - AIO',
        'DT-AIO': 'DT - AIO',
        'nan': 'NOTEBOOK',
        'TOWER': 'OTROS'
    }

    subfamilia_mapping_dict = {
        'PORTAFOLIO NB': 'PORTAFOLIO NB',
        'Portafolio Aio': 'AIO',
        'Portafolio TB': 'PORTAFOLIO TB',
        'Slim': 'PORTAFOLIO NB',
        'Gaming': 'GAMING',
        '2en1': '2_en_1',
        'SLIM': 'PORTAFOLIO NB',
        'Portafolio TOWER': 'OTROS',
        'Portafolio Tower': 'OTROS',
        'Portafolio AIO': 'AIO',
        'PORTAFOLIO AIO': 'AIO',
        '0': 'OTROS',
        'GAMING': 'GAMING'
    }

    df_precio['familia'] = df_precio['familia'].map(familia_mapping_dict)
    df_precio['subfamilia'] = df_precio['subfamilia'].map(subfamilia_mapping_dict)
    df_precio['Fecha_semanal'] = df_precio['fecha'] - pd.to_timedelta(df_precio['fecha'].dt.dayofweek, unit='d')
    df_precio.drop(columns=['fecha'], inplace=True)
    df_precio['Fecha_semanal'] = pd.to_datetime(df_precio['Fecha_semanal'])
    df_precio.rename(columns={'Fecha_semanal': 'fecha'}, inplace=True)

    df_precio = df_precio.groupby(['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia', 'modelo'])['prom_precio'].mean().reset_index()

    print("Lenovo prices preprocesed")
    return df_precio

def preprocessDolar(df_dolar):
    df_dolar['fecha'] = pd.to_datetime(df_dolar['fecha'])
    
    print("Dolar preprocesed")
    return df_dolar

def preprocessShareOfPeople(df_people, df_master_cadena):
    nuevas_filas = []

    df_people['fecha'] = pd.to_datetime(df_people['fecha'])

    df_people['year'] = df_people['fecha'].dt.year
    df_people['month'] = df_people['fecha'].dt.month
    df_people['month'] = pd.to_numeric(df_people['month'])

    df_people = df_people.groupby(['year', 'month', 'codigo_pdv'])['cant_shareofpeople'].sum().reset_index()
    df_people['fecha'] = pd.to_datetime(df_people[['year', 'month']].assign(DAY=1))
    df_people.drop(columns=['year', 'month'], inplace=True)

    for _, row in df_people.iterrows():
        lunes_del_mes = generar_lunes_del_mes(row['fecha'])
        for lunes in lunes_del_mes:
            nuevas_filas.append({
                'codigo_pdv': row['codigo_pdv'],
                'cant_shareofpeople': row['cant_shareofpeople'],
                'fecha_semana': lunes
            })

    df_people = pd.DataFrame(nuevas_filas).query('fecha_semana < "2024-08-05"')
    df_people.rename(columns={'fecha_semana': 'fecha'}, inplace=True)

    df_people = df_people.merge(df_master_cadena, how='left', on='codigo_pdv')
    print("Lenovo share of People preprocesed")
    return df_people

def preprocessMobiliario(df_mobiliario, df_master_cadena):
    df_mobiliario['fecha'] = pd.to_datetime(df_mobiliario['fecha'])
    df_mobiliario = df_mobiliario.merge(df_master_cadena, how='left', on='codigo_pdv')

    df_mobiliario = df_mobiliario.groupby([
        pd.Grouper(key='fecha', freq='W-MON'),
        'cadena',
        'nombre_pdv',
        'codigo_pdv',
    ]).apply(custom2_agg).reset_index()

    print("Lenovo mobiliario preprocesed")
    return df_mobiliario

def preprocessBranding(df_branding, df_master_cadena):
    df_branding['fecha'] = pd.to_datetime(df_branding['fecha'])
    df_branding = df_branding.merge(df_master_cadena, how='left', on='codigo_pdv')

    df_branding = df_branding.groupby([
        pd.Grouper(key='fecha', freq='W-MON'),
            'cadena',
            'nombre_pdv',
            'codigo_pdv',
        ]).apply(custom_agg).reset_index()

    print("Lenovo branding preprocesed")
    return df_branding

def preprocessShareOfShelf(df_shelf, df_master_cadena):
    df_shelf = df_shelf.merge(df_master_cadena, how='left', on='codigo_pdv')
    df_shelf['cant_notebook_exhibicion'] = df_shelf['cant_nb_exhibicion'] + df_shelf['cant_gaming_exhibicion']
    df_shelf['cant_exhibicion_total'] = df_shelf['cant_notebook_exhibicion'] + df_shelf['cant_tablets_exhibicion'] + df_shelf['cant_desktops_exhibicion']
    df_shelf.drop(columns=['cant_nb_exhibicion', 'cant_gaming_exhibicion', 'cant_notebook_exhibicion', 'cant_tablets_exhibicion', 'cant_desktops_exhibicion'], inplace=True)
    df_shelf['fecha'] = pd.to_datetime(df_shelf['fecha'])

    df_shelf = df_shelf.groupby([
    pd.Grouper(key='fecha', freq='W-MON'),
        'cadena',
        'nombre_pdv',
        'codigo_pdv'
    ]).agg({
        'cant_exhibicion_total': 'sum',
    }).reset_index()

    print("Lenovo share of shelf preprocesed")
    return df_shelf

def preprocessCampanas(df_campanas):
    df_campanas['fecha'] = pd.to_datetime(df_campanas['fecha'])

    # Create new rows for the previous 5 years
    new_rows = []
    for year_offset in range(1, 6):
        temp_df = df_campanas.copy()
        temp_df['fecha'] = temp_df['fecha'] - pd.DateOffset(years=year_offset)
        new_rows.append(temp_df)

    # Concatenate the original dataframe with the new rows
    df_extended = pd.concat([df_campanas] + new_rows, ignore_index=True)

    # Adjust all dates to be Mondays
    df_extended['fecha'] = df_extended['fecha'] + pd.offsets.Week(weekday=0)

    # Sort by date and reset index
    df_campanas = df_extended.sort_values(by='fecha').reset_index(drop=True)

    print("Lenovo campaigns preprocesed")
    return df_campanas

# ----------------- Rivals Data Preprocessing ----------------- #

def preprocessVentasRivals(df_ventas):
    df_ventas = df_ventas.sort_values(by='fecha')
    df_ventas = df_ventas.reset_index(drop=True)
    df_ventas['subfamilia'] = np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'NOTEBOOK'), 'PORTAFOLIO NB', np.where((df_ventas['subfamilia'] == '0') & (df_ventas['familia'] == 'AIO'), 'AIO', df_ventas['subfamilia']))

    marca_mapping_dict = {
        'ACER': 'ACER',
        'ASUS': 'ASUS',
        'HP': 'HP',
        'APPLE': 'APPLE',
        'DELL': 'DELL',
        'HUAWEI': 'HUAWEI',
        'MSI': 'MSI',
        'HP ': 'HP',
        'ADVANCE' : 'ADVANCE',
        'SAMSUNG' : 'SAMSUNG',
        'LG' : 'LG',
        'MIRAY' : 'MIRAY',
        'TOSHIBA' : 'TOSHIBA',
        'Asus ' : 'ASUS',
        'ALIENWARE' : 'ALIENWARE',
        'EVOO' : 'EVOO',
        'MICROSOFT' : 'MICROSOFT',
        'FASSIL' : 'FASSIL',
        'ASUS ' : 'ASUS',
        'CHUWI' : 'CHUWI',
        'XPG' : 'XPG',
        'PC-YA' : 'PC-YA',
        'MLOGIX' : 'MLOGIX',
        'HONOR' : 'HONOR',
        'PC-ONE' : 'PC-ONE',
        'Acer' : 'ACER',
        'Dell' : 'DELL',
        'LG                                                ' : 'LG',
        'Advance' : 'ADVANCE',
        'ALLDOCUBE' : 'ALLDOCUBE',
        'GATEWAY' : 'GATEWAY',
        'VASTEC' : 'VASTEC',
        'JUMPER TECH' : 'JUMPER TECH',
        'EPIK' : 'EPIK',
        'Asus' : 'ASUS',
        'HP                                                ' : 'HP',
        'MSI ' : 'MSI'
    }

    familia_mapping_dict = {
        'NOTEBOOK': 'NOTEBOOK',
        'Notebook': 'NOTEBOOK',
        'AIO': 'DT - AIO',
        'NOTEBOOk': 'NOTEBOOK',
        'Tablet': 'TABLET'
    }

    subfamilia_mapping_dict = {
        'GAMING': 'GAMING',
        'CONVENCIONAL': 'PORTAFOLIO NB',
        '2en1': '2_en_1',
        'AIO': 'AIO',
        'Gaming': 'GAMING',
        'Convencional': 'PORTAFOLIO NB',
        '2EN1': '2_en_1',
        'Premium': 'PORTAFOLIO TB',
        'Aio': 'AIO',
        'PORTAFOLIO NB': 'PORTAFOLIO NB',
    }
    #df_ventas['marca'] = df_ventas['marca'].str.strip().str.upper()
    #df_ventas['familia'] = df_ventas['familia'].str.strip().str.upper()
    #df_ventas['subfamilia'] = df_ventas['subfamilia'].str.strip()
    df_ventas['marca'] = df_ventas['marca'].map(marca_mapping_dict)
    df_ventas['familia'] = df_ventas['familia'].map(familia_mapping_dict)
    df_ventas['subfamilia'] = df_ventas['subfamilia'].map(subfamilia_mapping_dict)

    # Eliminar los registros que no están en la lista
    df_ventas = df_ventas[df_ventas['marca'].isin(lenovo_competencia_lista)].copy()
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
    df_ventas['cant_ventas'] = pd.to_numeric(df_ventas['cant_ventas'])

    print("Rivals sales preprocesed")
    return df_ventas

#def preprocessShareOfPeopleRivals(df_people, df_master_cadena):

def preprocessMobiliarioRivals(df_mobiliario, df_master_cadena):
    df_mobiliario['fecha'] = pd.to_datetime(df_mobiliario['fecha'])
    df_mobiliario = df_mobiliario[df_mobiliario['marca'].isin(lenovo_competencia_lista)].copy()
    df_mobiliario = df_mobiliario.merge(df_master_cadena, how='left', on='codigo_pdv')
    df_mobiliario = df_mobiliario.groupby([
    pd.Grouper(key='fecha', freq='W-MON'),
        'marca',
        'cadena',
        'nombre_pdv',
        'codigo_pdv',
    ]).apply(custom2_agg).reset_index()

    print("Rivals mobiliario preprocesed")
    return df_mobiliario

def preprocessBrandingRivals(df_branding, df_master_cadena):
    df_branding['fecha'] = pd.to_datetime(df_branding['fecha'])
    df_branding = df_branding[df_branding['marca'].isin(lenovo_competencia_lista)].copy()
    df_branding = df_branding.merge(df_master_cadena, how='left', on='codigo_pdv')
    df_branding = df_branding.groupby([
    pd.Grouper(key='fecha', freq='W-MON'),
        'marca',
        'cadena',
        'nombre_pdv',
        'codigo_pdv',
    ]).apply(custom_agg).reset_index()

    print("Rivals branding preprocesed")
    return df_branding

def preprocessShareOfShelfRivals(df_shelf, df_master_cadena):
    df_shelf = df_shelf[df_shelf['marca'].isin(lenovo_competencia_lista)].copy()
    df_shelf['cant_notebook_exhibicion'] = df_shelf['cant_nb_exhibicion'] + df_shelf['cant_gaming_exhibicion']
    df_shelf['cant_exhibicion_total'] = df_shelf['cant_notebook_exhibicion'] + df_shelf['cant_tablets_exhibicion'] + df_shelf['cant_desktops_exhibicion']
    df_shelf.drop(columns=['cant_nb_exhibicion', 'cant_gaming_exhibicion', 'cant_notebook_exhibicion', 'cant_tablets_exhibicion', 'cant_desktops_exhibicion'], inplace=True)
    df_shelf['fecha'] = pd.to_datetime(df_shelf['fecha'])

    df_shelf = df_shelf.merge(df_master_cadena, how='left', on='codigo_pdv')

    df_shelf = df_shelf.groupby([
    pd.Grouper(key='fecha', freq='W-MON'),
        'marca',
        'cadena',
        'nombre_pdv',
        'codigo_pdv'
    ]).agg({
        'cant_exhibicion_total': 'sum'
    }).reset_index()

    print("Rivals share of shelf preprocesed")
    return df_shelf

# ----------------- Lenovo Data Combined ----------------- #

def combine1(df_ventas, df_precio): # Combina ventas y precios
    df_complete1 = pd.merge(df_ventas,df_precio, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia', 'modelo'], how='outer')

    print("Lenovo combined 1 successfully")
    return df_complete1

def combine3(df_complete1, df_dolar): # Combina combine2 y dolar
    df_complete3 = pd.merge(df_complete1,df_dolar, on=['fecha'], how='outer')

    print("Lenovo combined 3 successfully")
    return df_complete3

def combine4(df_complete3, df_shareofpeople): # Combina combine3 y share of people
    df_complete4 = pd.merge(df_complete3,df_shareofpeople, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Lenovo combined 4 successfully")
    return df_complete4

def combine5(df_complete4, df_mobiliario): # Combina combine4 y mobiliario
    df_complete5 = pd.merge(df_complete4,df_mobiliario, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Lenovo combined 5 successfully")
    return df_complete5

def combine6(df_complete5, df_branding): # Combina combine5 y branding
    df_complete6 = pd.merge(df_complete5,df_branding, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Lenovo combined 6 successfully")
    return df_complete6

def combine7(df_complete6,df_shelf): # Combina combine6 y share of shelf
    df_complete7 = pd.merge(df_complete6,df_shelf, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Lenovo combined 7 successfully")
    return df_complete7

def combine8(df_complete7,df_campanas): # Combina combine7 y campañas
    df_complete8 = pd.merge(df_complete7,df_campanas, on=['fecha'], how='outer')

    print("Lenovo combined 8 successfully")
    return df_complete8


# ----------------- Rivals Data Combined ----------------- #

def rivalsCombine1(df_ventas, df_mobiliario):
    df_complete1 = pd.merge(df_ventas,df_mobiliario, on=['fecha', 'marca', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Rivals combined 1 successfully")
    return df_complete1

def rivalsCombine2(df_complete1, df_branding):
    df_complete2 = pd.merge(df_complete1,df_branding, on=['fecha', 'marca', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Rivals combined 2 successfully")
    return df_complete2

def rivalsCombine3(df_complete2, df_shelf):
    df_complete3 = pd.merge(df_complete2,df_shelf, on=['fecha', 'marca', 'cadena', 'nombre_pdv', 'codigo_pdv'], how='outer')

    print("Rivals combined 3 successfully")
    return df_complete3

def transform_by_brand(df):
    # Rellenar los nulos de las columnas 'familia' y 'subfamilia' con un valor específico para tratarlos como categorías
    df['familia'] = df['familia'].fillna('NULL')
    df['subfamilia'] = df['subfamilia'].fillna('NULL')

    # Convertir columnas numéricas a tipo float (manejar nulos)
    numeric_columns = ['cant_ventas', 'prom_precio', 'cant_mobiliario_total', 'cant_branding_total', 'cant_exhibicion_total']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Obtener marcas únicas
    unique_brands = df['marca'].unique()

    # Inicializar DataFrame vacío
    transformed_df = pd.DataFrame()

    # Iterar sobre cada marca y transformar los datos
    for brand in unique_brands:
        brand_df = df[df['marca'] == brand].copy()
        brand_features = brand_df.drop(columns=['marca'])
        
        # Asegurarse de que no haya duplicados por medio de agregación (usando la media aquí)
        brand_features = brand_features.groupby(['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia']).mean().reset_index()

        # Renombrar las columnas para incluir el prefijo de la marca
        brand_features.columns = [f"{brand}_{col}" if col not in ['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia'] 
                                  else col for col in brand_features.columns]
        
        # Combinar los datos con el DataFrame transformado
        if transformed_df.empty:
            transformed_df = brand_features
        else:
            transformed_df = pd.merge(transformed_df, brand_features, 
                                      on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia'], 
                                      how='outer')

    transformed_df.sort_values(by='fecha', ascending=True)

    print("Rivals transformecion successfully")
    return transformed_df

# ----------------- Main Functions ----------------- #

# Defining main function
def extract_data_aws():
    # Load Lenovo data
    df_ventas = loadVentas(engine)
    df_precio_base = loadPreciosAnalista(engine)
    df_master_cadena = loadMasterCadena(engine)
    df_dolar = loadDolar(engine)
    df_shareofpeople = loadShareOfProple(engine)
    df_mobiliario = loadMobiliario(engine)
    df_branding = loadBrading(engine)
    df_shareofshelf = loadShareOfShelf(engine)
    df_campanas = loadCampanas(engine)

    print("Lenovo Data loaded successfully")

    df_ventas = preprocessVentas(df_ventas)
    df_precio_base = preprocessPrecio(df_precio_base)
    df_dolar = preprocessDolar(df_dolar)
    df_shareofpeople = preprocessShareOfPeople(df_shareofpeople, df_master_cadena)
    df_mobiliario = preprocessMobiliario(df_mobiliario, df_master_cadena)
    df_branding = preprocessBranding(df_branding, df_master_cadena)
    df_shareofshelf = preprocessShareOfShelf(df_shareofshelf, df_master_cadena)
    df_campanas = preprocessCampanas(df_campanas)

    print("Lenovo Data preprocessing successfully")
    
    df_complete1 = combine1(df_ventas, df_precio_base)
    df_complete3 = combine3(df_complete1, df_dolar)
    df_complete4 = combine4(df_complete3, df_shareofpeople)
    df_complete5 = combine5(df_complete4, df_mobiliario)
    df_complete6 = combine6(df_complete5, df_branding)
    df_complete7 = combine7(df_complete6, df_shareofshelf)
    raw_final_data = combine8(df_complete7, df_campanas)
    
    print("Lenovo Tables combined successfully")

    # Load Rivals Data
    df_ventas_rivals = ld.loadRivalsVentas(engine)
    df_mobiliario_rivals = ld.loadRivalsMobiliario(engine)
    df_branding_rivals = ld.loadRivalsBranding(engine)
    df_shareofshelf_rivals = ld.loadRivalsShareOfShelf(engine)

    print("Rivals Data loaded successfully")

    df_ventas_rivals = preprocessVentasRivals(df_ventas_rivals)
    df_mobiliario_rivals = preprocessMobiliarioRivals(df_mobiliario_rivals, df_master_cadena)
    df_branding_rivals = preprocessBrandingRivals(df_branding_rivals, df_master_cadena)
    df_shareofshelf_rivals = preprocessShareOfShelfRivals(df_shareofshelf_rivals, df_master_cadena)

    print("Rivals Data preprocessing successfully")

    rivals_complete1 = rivalsCombine1(df_ventas_rivals, df_mobiliario_rivals)
    rivals_complete2 = rivalsCombine2(rivals_complete1, df_branding_rivals)
    rivals_complete3 = rivalsCombine3(rivals_complete2, df_shareofshelf_rivals)

    df_competencia = transform_by_brand(rivals_complete3)

    print("Rivals Tables combined successfully")

    # Combine Lenovo with rivals (Acer, Asus, HP, etc)
    table_final_combined = pd.merge(raw_final_data,df_competencia, on=['fecha', 'cadena', 'nombre_pdv', 'codigo_pdv', 'familia', 'subfamilia'], how='outer')

    print("All Tables combined successfully, Done!")

    return table_final_combined


def imputar_datos_por_tienda_anio(data, umbral=0.40, n_neighbors=5):
    """
    Imputa valores faltantes en el DataFrame `data` usando el método K-Nearest Neighbors (KNN) 
    por cada combinación de `codigo_tienda` y `año`. La imputación solo se aplica a las columnas 
    con al menos un umbral de datos completos definido.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene las columnas `fecha`, `codigo_tienda`, y los datos a imputar.
    - umbral (float): Proporción mínima de datos completos en una columna para permitir la imputación (default 0.60).
    - n_neighbors (int): Número de vecinos a considerar en la imputación KNN (default 5).

    Returns:
    - pd.DataFrame: DataFrame con valores faltantes imputados según el método KNN.
    """
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['año'] = data['fecha'].dt.year

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = KNNImputer(n_neighbors=n_neighbors)

    for codigo_tienda in data['codigo_tienda'].unique():
        for año in data['año'].unique():
            df_tienda_año = data[(data['codigo_tienda'] == codigo_tienda) & (data['año'] == año)]
            
            for columna in df_tienda_año.columns[df_tienda_año.isna().any()]:
                non_null_percentage = df_tienda_año[columna].notna().mean()
                
                total_filas = len(df_tienda_año)
                valores_completos = df_tienda_año[columna].notna().sum()
                
                # Imputar solo si el porcentaje de datos completos es mayor o igual al umbral
                if non_null_percentage >= umbral:

                    print(f"Tienda: {codigo_tienda}, Año: {año}, Columna: {columna}")
                    print(f"Total de filas: {total_filas}, Valores completos antes de imputación: {valores_completos}")
                    
                    df_temp = df_tienda_año[[columna]]
                    
                    data.loc[df_tienda_año.index, columna] = imputer.fit_transform(df_temp)
                    
                    print(f"Imputación realizada para tienda {codigo_tienda}, año {año}, columna {columna}.")
                else:
                    print(f"No se realizó imputación para tienda {codigo_tienda}, año {año}, columna {columna}, ya que no cumple con el umbral de {umbral*100}% de datos presentes.")
    
    return data


def limpieza_features(data):
    """
    Realiza una limpieza y transformación de características en el DataFrame `data`.
    
    Esta función combina algunas columnas en una nueva columna única, realiza la codificación 
    de variables categóricas con Label Encoding y filtra el conjunto de datos según 
    ciertos criterios de completitud y frecuencia.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene las columnas `codigo_tienda`, `familia`, `subfamilia`, 
      `nombre_campana`, `tipo_cadena`, `HP_cant_ventas`, `ASUS_cant_ventas`, `ACER_cant_ventas`, `cant_ventas` y `fecha`.

    Returns:
    - pd.DataFrame: DataFrame con las características transformadas, sin valores nulos en columnas seleccionadas
      y filtrado para incluir solo los valores frecuentes en `codigo_tienda_familia`.
    """
    
    data['codigo_tienda_familia'] = data['codigo_tienda'] + '_' + data['familia'] + '_' + data['subfamilia']
    
    data.drop(['codigo_tienda', 'familia', 'subfamilia'], axis=1, inplace=True)
    
    label_encoder = LabelEncoder()
    data['nombre_campana'] = label_encoder.fit_transform(data['nombre_campana'])
    data['tipo_cadena'] = label_encoder.fit_transform(data['tipo_cadena'])
    
    # Eliminar filas que tengan valores nulos en las columnas de ventas seleccionadas
    data = data.dropna(subset=['HP_cant_ventas', 'ASUS_cant_ventas', 'ACER_cant_ventas', 'cant_ventas'])
    
    # Filtrar para incluir solo los valores de 'codigo_tienda_familia' con más de 100 ocurrencias
    data_util = data[data["codigo_tienda_familia"].isin(data["codigo_tienda_familia"].value_counts()[data["codigo_tienda_familia"].value_counts() > 100].index)]
    
    data_util['fecha'] = pd.to_datetime(data_util['fecha'])
    
    return data_util

def cortar_data_por_fecha(data, train_start='2019-01-01', train_end='2023-12-31', test_start='2024-01-01', test_end='2024-12-31'):
    """
    Divide el DataFrame `data` en conjuntos de entrenamiento y prueba basados en fechas y los guarda en archivos CSV
    dentro de la carpeta `data`.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene la columna 'fecha' con datos en formato datetime.
    - train_start (str): Fecha de inicio del conjunto de entrenamiento en formato 'YYYY-MM-DD' (default '2019-01-01').
    - train_end (str): Fecha de fin del conjunto de entrenamiento en formato 'YYYY-MM-DD' (default '2023-12-31').
    - test_start (str): Fecha de inicio del conjunto de prueba en formato 'YYYY-MM-DD' (default '2024-01-01').
    - test_end (str): Fecha de fin del conjunto de prueba en formato 'YYYY-MM-DD' (default '2024-12-31').

    Returns:
    - None: La función guarda los archivos `train.csv` y `test.csv` en la carpeta `data`.
    """
    data['fecha'] = pd.to_datetime(data['fecha'])

    train = data[(data['fecha'] >= train_start) & (data['fecha'] <= train_end)]
    test = data[(data['fecha'] >= test_start) & (data['fecha'] <= test_end)]

    output_dir = os.path.join(os.path.dirname(__file__), '../../data')
    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Conjunto de entrenamiento guardado en '{output_dir}/train.csv'")
    print(f"Conjunto de prueba guardado en '{output_dir}/test.csv'")

    #TODO HACER TODO EN UNO SOLO. 

def lectura_de_datos_csv():
    NOMBRE_CSV = os.getenv("NOMBRE_CSV")
    data = pd.read_csv(NOMBRE_CSV)
    return data

def process_data_from_aws(path="src"):
    print("Obteniendo data desde AWS...")
    try:
        data = extract_data_aws()
        #data.to_csv(os.path.join(path, "data.csv"), index=False)
        return data
    except Exception as e:
        print(f"Error al obtener la data desde AWS: {e}")
        return None
    #
    print("Información guardada en 'data_combined_test.csv")
    #process_data(data)

def process_data_from_csv():
    print("Obteniendo data desde un archivo CSV...")
    data = lectura_de_datos_csv()
    process_data(data) # dataTable_M12_S1

def process_data(data):
    print("Proceso de imputación")
    df_imputada = imputar_datos_por_tienda_anio(data)
    print("Proceso de limpieza")
    df_imputada = limpieza_features(df_imputada)
    df_imputada.to_csv("data_preprocessed_test.csv", index=False)
    print("Proceso de corte")
    #cortar_data_por_fecha(df_imputada)

