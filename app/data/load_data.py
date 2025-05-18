import logging.config
import time
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy.engine import create_engine

# ----------------- Lenovo Data ----------------- #

def loadVentas(engine):
    query = f'''
    SELECT DATE(fecha) AS fecha, cadena, nombre_pdv, codigo_pdv_b2b AS codigo_pdv, categoria1 AS familia, cluster_pdv AS subfamilia, categoria4 AS modelo, SUM(ABS(vtas_unidades)) AS cant_ventas, AVG(ABS(precio)) AS prom_precio_b2b
    FROM dim_master_b2b_cliente_id_330
    WHERE DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), cadena, nombre_pdv, codigo_pdv_b2b, categoria1, cluster_pdv, categoria4
    ORDER BY fecha DESC
    '''
    
    conn = engine.connect()
    df_ventas = pd.read_sql_query(query, conn)
    print("Lenovo sales loaded")
    return df_ventas

def loadMasterCadena(engine):
    query = f'''
    SELECT cadena, nombre_pdv, codigo_pdv_b2b AS codigo_pdv, COUNT(*) AS cant_registros
    FROM dim_master_b2b_cliente_id_330
    GROUP BY cadena, nombre_pdv, codigo_pdv_b2b
    '''

    conn = engine.connect()
    df_master_cadena = pd.read_sql_query(query, conn)
    df_master_cadena.drop(columns=['cant_registros'], inplace=True)
    print("Lenovo master cadena loaded")
    return df_master_cadena

def loadPreciosAnalista(engine):
    query_precio = f'''
    SELECT DATE(fecha) AS fecha, cadena, nombre_pdv, codigo_pdv AS codigo_pdv, categoria2 AS familia, categoria AS subfamilia, modelo, AVG(precio_final) AS prom_precio
    FROM dim_lenovo_base_precios_cliente_id_330
    WHERE (marca = 'Lenovo' OR marca = 'LENOVO') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), cadena, nombre_pdv, codigo_pdv, categoria2, categoria, modelo
    ORDER BY fecha DESC
    '''

    conn = engine.connect()
    df_precio = pd.read_sql_query(query_precio, conn)
    print("Lenovo prices loaded")
    return df_precio

def loadDolar(engine):
    query_dolar = f'''
    SELECT date_trunc('week', DATE(date)) AS fecha, AVG(price) AS prom_dolar
    FROM dim_usd_pen_historical
    WHERE date_trunc('week', DATE(date)) >= DATE('2019-01-07')
    GROUP BY date_trunc('week', DATE(date))
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_dolar = pd.read_sql_query(query_dolar, conn)
    print("Dolar loaded")
    return df_dolar

def loadShareOfProple(engine):
    query_people = f'''
    SELECT DATE(fecha) AS fecha, codtienda AS codigo_pdv, COUNT(*) AS cant_shareofpeople
    FROM dim_lenovo_planning_cliente_id_330
    WHERE DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codtienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_people = pd.read_sql_query(query_people, conn)
    print("Lenovo share of people loaded")
    return df_people

def loadMobiliario(engine):
    query_mobiliario = f'''
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, SUM(COALESCE(mobiliario_notebook,0) + COALESCE(mobiliario_tablet,0) + COALESCE(mobiliario_gaming,0) + COALESCE(mobiliario_multicategoria,0)) AS cant_mobiliario_total
    FROM dim_lenovo_mobil_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_mobiliario = pd.read_sql_query(query_mobiliario, conn)
    print("Lenovo Mobiliario loaded")
    return df_mobiliario

def loadBrading(engine):
    query_branding = f'''
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, SUM(COALESCE(cubre_sensores,0) + COALESCE(wall,0) + COALESCE(cajas_luz,0) + COALESCE(intervenciones, 0) + COALESCE(columnas_branding_exhibicion, 0) + COALESCE(columnas_branding, 0) + COALESCE(glorificadores_marca,0)) AS cant_branding_total
    FROM dim_lenovo_branding_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_branding = pd.read_sql_query(query_branding, conn)
    print("Lenovo Branding loaded")
    return df_branding

def loadShareOfShelf(engine):
    query_shelf = f'''
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, SUM(COALESCE(tablets_exhibicion,0)) AS cant_tablets_exhibicion, SUM(COALESCE(notebooks_exhibicion,0)) AS cant_nb_exhibicion, SUM(COALESCE(desktops_exhibicion,0)) AS cant_desktops_exhibicion
    FROM dim_lenovo_shelf_nb_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    UNION
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, SUM(COALESCE(exhibiciones,0)) AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_shelf_gam_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    UNION
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND familia = 'NOTEBOOK' AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    UNION
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND familia = 'TABLET' AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    UNION
    SELECT DATE(fecha) AS fecha, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca = 'Lenovo' OR nombre_marca = 'LENOVO') AND familia = 'AIO' AND DATE(fecha) < DATE('2024-08-05') AND DATE(fecha) >= DATE('2019-01-07')
    GROUP BY DATE(fecha), codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_shelf = pd.read_sql_query(query_shelf, conn)
    print("Lenovo Share of Shelf loaded")
    return df_shelf

def loadCampanas(engine):
    query_campanas = f'''
    SELECT DATE(start_date) AS fecha, campana AS nombre_campana
    FROM dim_lenovo_campana_cliente_id_330
    WHERE DATE(start_date) >= DATE('2019-01-07')
    '''
    conn = engine.connect()
    df_campanas = pd.read_sql_query(query_campanas, conn)
    print("Lenovo campaigns loaded")
    return df_campanas

# ----------------- Rivals Data ----------------- #

def loadRivalsVentas(engine):
    query = f'''
    SELECT DATE(semana_dia) AS fecha, marca, retail AS cadena, sucursal AS nombre_pdv, codtienda AS codigo_pdv, familia, sub_familia AS subfamilia, SUM(ABS(unidades)) AS cant_ventas, AVG(ABS(venta)/ABS(unidades)) AS prom_precio
    FROM dim_lenovo_retail_oneview_cliente_id_330
    WHERE (marca != 'Lenovo' AND marca != 'LENOVO' AND marca != 'LENOVO ') AND DATE(semana_dia) < DATE('2024-08-05')
    GROUP BY semana_dia, marca, retail, sucursal, codtienda, familia, sub_familia
    ORDER BY semana_dia DESC
    '''
    
    conn = engine.connect()
    df_ventas_base = pd.read_sql_query(query, conn)
    print("Rivals sales loaded")
    return df_ventas_base

def loadRivalsMobiliario(engine):
    query_mobiliario = f'''
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, SUM(COALESCE(mobiliario_notebook,0) + COALESCE(mobiliario_tablet,0) + COALESCE(mobiliario_gaming,0) + COALESCE(mobiliario_multicategoria,0)) AS cant_mobiliario_total
    FROM dim_lenovo_mobil_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY DATE(fecha), nombre_marca, codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_mobiliario = pd.read_sql_query(query_mobiliario, conn)
    print("Rivals Mobiliario loaded")
    return df_mobiliario

def loadRivalsBranding(engine):
    query_branding = f'''
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, SUM(COALESCE(cubre_sensores,0) + COALESCE(wall,0) + COALESCE(cajas_luz,0) + COALESCE(intervenciones, 0) + COALESCE(columnas_branding_exhibicion, 0) + COALESCE(columnas_branding, 0) + COALESCE(glorificadores_marca,0)) AS cant_branding_total
    FROM dim_lenovo_branding_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY DATE(fecha), nombre_marca, codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_branding = pd.read_sql_query(query_branding, conn)
    print("Rivals Branding loaded")
    return df_branding

def loadRivalsShareOfShelf(engine):
    query_shelf = f'''
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, SUM(COALESCE(tablets_exhibicion,0)) AS cant_tablets_exhibicion, SUM(COALESCE(notebooks_exhibicion,0)) AS cant_nb_exhibicion, SUM(COALESCE(desktops_exhibicion,0)) AS cant_desktops_exhibicion
    FROM dim_lenovo_shelf_nb_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY fecha, nombre_marca, bandeira, codTienda
    UNION
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, SUM(COALESCE(exhibiciones,0)) AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_shelf_gam_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY fecha, nombre_marca, bandeira, codTienda
    UNION
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND familia = 'NOTEBOOK' AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY fecha, nombre_marca, tienda, codTienda
    UNION
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, 0 AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND familia = 'TABLET' AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY fecha, nombre_marca, tienda, codTienda
    UNION
    SELECT DATE(fecha) AS fecha, nombre_marca AS marca, codTienda AS codigo_pdv, 0 AS cant_gaming_exhibicion, 0 AS cant_tablets_exhibicion, 0 AS cant_nb_exhibicion, SUM(COALESCE(cantidad,0)) AS cant_desktops_exhibicion
    FROM dim_lenovo_display_share_cliente_id_330
    WHERE (nombre_marca != 'Lenovo' AND nombre_marca != 'LENOVO' AND nombre_marca != 'LENOVO ') AND familia = 'AIO' AND DATE(fecha) < DATE('2024-08-05')
    GROUP BY fecha, nombre_marca, tienda, codTienda
    ORDER BY fecha ASC
    '''
    conn = engine.connect()
    df_shelf = pd.read_sql_query(query_shelf, conn)
    print("Rivals Share of Shelf loaded")
    return df_shelf