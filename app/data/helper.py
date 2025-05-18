import pandas as pd

def generar_lunes_del_mes(fecha):
    # Obtener el primer día y el último día del mes
    primer_dia_mes = fecha.replace(day=1)
    ultimo_dia_mes = (primer_dia_mes + pd.offsets.MonthEnd(1)).normalize()
    
    # Generar todas las fechas del mes y filtrar solo los lunes
    lunes = pd.date_range(start=primer_dia_mes, end=ultimo_dia_mes, freq='W-MON')
    
    return lunes


def custom2_agg(group):
    # Agregar las columnas numéricas
    MobiliarioTotalMarca = group['cant_mobiliario_total'].sum()
    
    # Retornar los resultados como una Serie
    return pd.Series({
        'cant_mobiliario_total': MobiliarioTotalMarca
})

def custom_agg(group):
    cant_branding_total = group['cant_branding_total'].sum()
    
    # Retornar los resultados como una Serie
    return pd.Series({
        'cant_branding_total': cant_branding_total
    })