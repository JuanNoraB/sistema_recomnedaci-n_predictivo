#%%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Data"
HISTORICAL_FILE = DATA_DIR / "Historico_08122025.csv"
OUTPUT_FILE = BASE_DIR / "features_final_all_familias.csv"

RAW_DTYPES = {
    "COD_SUBCATEGORIA": "Int64",
    "COD_CATEGORIA": "Int64",
    "COD_UNIDAD_COMERCIAL": "Int64",
    "COD_ITEM": "Int64",
    "DIM_FACTURA": "Int64",
    "COD_LOCAL": "Int64",
    "CODIGO_FAMILIA": "Int64",
}

NUMERIC_COLUMNS = ["CANTIDAD_SUELTA", "PVP", "VENTA_NETA", "DESCUENTO"]
PARSE_DATES = ["DIM_PERIODO"]

CUTOFF_DATE = pd.Timestamp("2025-11-30")  # Actualizado para incluir datos de noviembre
RECENT_WINDOW_DAYS = 60
FREQUENCY_WINDOW_DAYS = 180
SOW_MONTHS_24 = 24
SOW_MONTHS_12 = 12


def load_historical_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo histórico en: {path}")

    df = pd.read_csv(
        path,
        dtype=RAW_DTYPES,  #indicar el tipo de dato por columna
        parse_dates=PARSE_DATES, #indicar las columnas que son fechas
        dayfirst=True,
        infer_datetime_format=True, #intenta adividar el formato de fecha y optimizar el parseeo
        encoding='latin-1'
    )

    df["DIM_PERIODO"] = pd.to_datetime(
        df["DIM_PERIODO"],
        format="%d-%b-%y", # ejemplo: 24-Nov-25, para 25-11-2025 SERIA "format = %d-%m-%Y"
        errors="coerce",
    )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)

    #filtrar solo familia especifica 1719363028
    #ceduleas  = [1102088158,1719363028]
    #df = df[df["CODIGO_FAMILIA"].isin(ceduleas)]


    return df

def penalize_outliers(data, std_threshold):
    arr = np.asarray(data, dtype=float)
    if arr.size < 2: #quitar esta mierda que esta demas 
        return arr.tolist(), [2,0,0] # Devuelve lista y fuerza de castigo
    
    mean, std = np.mean(arr), np.std(arr)
    if std == 0:
        return arr.tolist(), [2,0,0]

    # Calculamos la "fuerza" como la proporción de la std sobre la media
    # Esto nos da una idea de qué tan grande es la dispersión
    fuerza_castigo = std / mean if mean > 0 else 1.0
    
    upper_bound = mean + std_threshold * std
    lower_bound = mean - std_threshold * std
    penalized_arr = np.clip(arr, lower_bound, upper_bound)
    
    return penalized_arr.tolist(), [fuerza_castigo, mean, std]

def calcular_ciclos_por_bloques(
    df_ventas,
    familia_id,
    meses_historico=14,
    periodo_dias=7,
    min_bloques_compra=3,
    max_compras_recientes=10, # Límite de compras a usar
    std_threshold=0.1, # para el suavizado de los ciclossss 
    today=pd.Timestamp.today(),
):
    today = today.normalize() # 0s a la hora actual
    fecha_inicio = today - pd.DateOffset(months=meses_historico)

    df_fam = df_ventas[
        (df_ventas["CODIGO_FAMILIA"] == familia_id) &
        (pd.to_datetime(df_ventas["DIM_PERIODO"]) >= fecha_inicio)
    ].copy()

    if df_fam.empty:
        return pd.DataFrame()

    dias_desde_inicio = (df_fam["DIM_PERIODO"] - fecha_inicio).dt.days #dias que pasaron desde la fecha del inicio 
    df_fam["bloque"] = dias_desde_inicio // periodo_dias

    resultados = []

    for subcat, df_sub in df_fam.groupby("COD_SUBCATEGORIA"):
        # Ordenamos por fecha para tomar las más recientes
        df_sub = df_sub.sort_values("DIM_PERIODO", ascending=False)
        bloques_con_compra = np.sort(df_sub["bloque"].unique())

        # Usamos solo las N compras más recientes
        if len(bloques_con_compra) > max_compras_recientes:
            bloques_con_compra = bloques_con_compra[-max_compras_recientes:]

        # Si no cumple el mínimo, retornamos valores por defecto
        if len(bloques_con_compra) < min_bloques_compra:
            resultados.append({
                "CODIGO_FAMILIA": familia_id, "COD_SUBCATEGORIA": subcat,
                "ciclo_dias": 0, "fuerza_castigo": [2,0,0],
                "gaps_originales_dias": [], "gaps_suavizados_dias": []
            })
            continue

        gaps_bloques = np.diff(bloques_con_compra)
        if len(gaps_bloques) == 0: # solo en el caso de que numero de compras se setee en 1 pero esto nlv se va hacer
            continue
            
        gaps_dias = gaps_bloques * periodo_dias
        
        gaps_suavizados, fuerza_castigo = penalize_outliers(gaps_dias, std_threshold=std_threshold)
        ciclo_dias = np.mean(gaps_suavizados)
        ciclo_dias_sin_suavisado = np.mean(gaps_dias)

        resultados.append({
            "CODIGO_FAMILIA": familia_id, "COD_SUBCATEGORIA": subcat,
            "ciclo_dias": ciclo_dias, "ciclo_dias_sin_suavisado": ciclo_dias_sin_suavisado, "fuerza_castigo": fuerza_castigo,
            "gaps_originales_dias": gaps_dias.tolist(), "gaps_suavizados_dias": gaps_suavizados
        })

    df_resultado = pd.DataFrame(resultados)
    
    # Ordenar para que los que no tienen ciclo queden al final
    if not df_resultado.empty:
        df_resultado = df_resultado.sort_values("fuerza_castigo", ascending=True).reset_index(drop=True)

    return df_resultado
    

def compute_recency_features(subcat_agg: pd.DataFrame,
                             ciclos_estacionales: pd.DataFrame,
                             fecha_corte: pd.Timestamp) -> pd.DataFrame:
    subcat_agg = subcat_agg.copy()

    # días desde última compra
    subcat_agg["dias_desde_ultima_compra"] = (
        fecha_corte - subcat_agg["ultima_compra"]
    ).dt.days.clip(lower=0)

    # pegar ciclo_dias
    subcat_agg = subcat_agg.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")

    mask_dias = subcat_agg["ciclo_dias"] > 0.0

    # recencia half-life base
    subcat_agg["recencia"] = 0.0
    subcat_agg.loc[mask_dias, "recencia"] = 1 - np.power(
        2,
        - subcat_agg.loc[mask_dias, "dias_desde_ultima_compra"].to_numpy()
          / subcat_agg.loc[mask_dias, "ciclo_dias"].to_numpy()
    )
    subcat_agg.loc[mask_dias, "recencia"] = np.minimum(1.0,2.0*subcat_agg.loc[mask_dias, "recencia"])

    # --- castigo por N ciclos sin compra (exponencial curva) ---
    r = (
        subcat_agg.loc[mask_dias, "dias_desde_ultima_compra"]
        / subcat_agg.loc[mask_dias, "ciclo_dias"]
    )

    subcat_agg.loc[mask_dias, "l_compra_sobre_ciclo"] = r
    r = np.maximum(1.0, r)  # a partir de 1 ciclo

    k = 0.1
    p = 2.5
    castigo = np.exp(-k * np.power(r - 1, p))  # 1 →1, 2→0.9, 3→0.57, 4→0.21, 5→0.04

    subcat_agg["castigo_recencia"] = 0.0
    subcat_agg.loc[mask_dias, "castigo_recencia"] = castigo

    # combinar recencia HL * castigo
    subcat_agg.loc[mask_dias, "recencia_hl"] = (
        subcat_agg.loc[mask_dias, "recencia"]
        * subcat_agg.loc[mask_dias, "castigo_recencia"]
    )

    return subcat_agg[["COD_SUBCATEGORIA", "recencia_hl","castigo_recencia","l_compra_sobre_ciclo","dias_desde_ultima_compra","recencia"]]


def compute_frequency_features(df_family: pd.DataFrame,ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp,periodo_dias: int = 7) -> pd.DataFrame:
    ventana_inicio = fecha_corte - pd.Timedelta(days=FREQUENCY_WINDOW_DAYS)
    recientes = df_family[df_family["DIM_PERIODO"] >= ventana_inicio].copy()

    recientes['mes']= recientes['DIM_PERIODO'].dt.month

    #dias_desde_inicio = (recientes["DIM_PERIODO"] - ventana_inicio).dt.days
    #recientes["bloque"] = dias_desde_inicio // periodo_dias

    freq = (
        recientes.groupby(["COD_SUBCATEGORIA"])
        .agg(
            compras=("DIM_FACTURA", "count")
        )
        .reset_index()
    )
    # 10 
    if freq.empty:
        return pd.DataFrame({"COD_SUBCATEGORIA": df_family["COD_SUBCATEGORIA"].unique(), "freq_score": 0.1})


    freq_ciclo = freq.merge(ciclos_estacionales[["COD_SUBCATEGORIA", "ciclo_dias"]], on="COD_SUBCATEGORIA", how="left")
    # 180 / 30 = 6  compras cada 30 dias uno 
    # 1) avg_compras: 0 cuando ciclo_dias == 0, si no, 90 / ciclo_dias (o lo que sea FREQUENCY_WINDOW_DAYS)
    #15 7 dias   6  3 meses      
    freq_ciclo["avg_compras"] = 0.0
    mask_ciclo_pos = freq_ciclo["ciclo_dias"] > 0

    freq_ciclo.loc[mask_ciclo_pos, "avg_compras"] = np.round(
        (FREQUENCY_WINDOW_DAYS / freq_ciclo.loc[mask_ciclo_pos, "ciclo_dias"]) * 1.2
    )

    # 2) freq_score: 0 cuando avg_compras == 0,
    freq_ciclo["freq_score"] = 0.0
    mask_avg_pos = freq_ciclo["avg_compras"] > 0

    # --- NUEVA CURVA DE FRECUENCIA ---

    alpha = 0.25     # curva para r < 1
    base_min = 0.4   # valor en r = 1
    base_max = 1.0   # valor en r = 0
    r_max = 1.6      # en r = 1.6 el score cae a 0

    # ratio compras / promedio
    ratio = (
        freq_ciclo.loc[mask_avg_pos, "compras"] /
        freq_ciclo.loc[mask_avg_pos, "avg_compras"]
    ).to_numpy()

    score = np.zeros_like(ratio)
    freq_ciclo["ratio"] = 10
    freq_ciclo.loc[mask_avg_pos, "ratio"] = ratio.astype(float)
    # tramo r <= 1
    mask_low = ratio <= 1
    raw_low = (1 - ratio[mask_low]) ** alpha          # 1 en r=0, 0 en r=1
    score[mask_low] = base_min + (base_max - base_min) * raw_low
    # => r=0 -> 1 ; r=1 -> 0.4

    # tramo 1 < r <= r_max : cola lineal de 0.4 a 0
    mask_high = ratio > 1
    exceso = np.minimum(ratio[mask_high], r_max) - 1   # 0 en r=1, r_max-1 en r_max
    score[mask_high] = base_min * np.maximum(
        0,
        1 - exceso / (r_max - 1)
    )
    # ratio > r_max → 0

    # guardamos en freq_score SOLO para los que tienen avg_compras>0
    freq_ciclo.loc[mask_avg_pos, "freq_score"] = score

    return freq_ciclo[["COD_SUBCATEGORIA", "freq_score","avg_compras","compras","ratio"]]


def compute_sow_features(df_family: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:

    ventana_12m_inicio = fecha_corte - pd.DateOffset(months=SOW_MONTHS_12)
    ventana_24m_inicio = fecha_corte - pd.DateOffset(months=SOW_MONTHS_24)

    # Últimos 12 meses
    historicos_12m = df_family[df_family["DIM_PERIODO"] >= ventana_12m_inicio]

    # De 24 a 12 meses atrás
    historicos_24m = df_family[
        (df_family["DIM_PERIODO"] >= ventana_24m_inicio)
        & (df_family["DIM_PERIODO"] < ventana_12m_inicio)
    ]

    # --- 12 meses: transacciones por subcategoría ---
    sow_agg_12m = (
        historicos_12m.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("DIM_FACTURA", "count"))
        .reset_index()
    )

    # --- 24–12 meses: transacciones por subcategoría ---
    sow_agg_24m = (
        historicos_24m.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("DIM_FACTURA", "count"))
        .reset_index()
    )

    # Pesos: más reciente (12m) *7, más antiguo (24–12) *3
    sow_agg_12m["transacciones_netas"] = sow_agg_12m["transacciones_netas"] * 7
    sow_agg_24m["transacciones_netas"] = sow_agg_24m["transacciones_netas"] * 3

    # Unión vertical (no merge)
    sow_agg = pd.concat([sow_agg_12m, sow_agg_24m], ignore_index=True)

    if sow_agg.empty:
        # Si no hay datos en ninguna ventana, devolver todo en 0
        return pd.DataFrame(
            {
                "COD_SUBCATEGORIA": df_family["COD_SUBCATEGORIA"].unique(),
                "sow_24m": 0.0,
            }
        )

    # Sumar pesos por subcategoría
    sow_agg = (
        sow_agg.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("transacciones_netas", "sum"))
        .reset_index()
    )

    total_transacciones = sow_agg["transacciones_netas"].sum()

    if total_transacciones <= 0:
        sow_agg["sow_24m"] = 0.0
    else:
        sow_agg["sow_24m"] = sow_agg["transacciones_netas"] / total_transacciones

    max_sow_24m = sow_agg["sow_24m"].max()
    mutiplicador = 1 / max_sow_24m
    sow_agg["sow_24m"] = sow_agg["sow_24m"] * mutiplicador

    return sow_agg[["COD_SUBCATEGORIA", "sow_24m","transacciones_netas"]]


def _discretize_series(series, umbral):
    """
    Convierte una serie de transacciones en una serie binaria (0 o 1)
    usando la suma de meses adyacentes.
    """
    discretized = np.zeros(len(series), dtype=int)
    for i in range(len(series) - 1):
        # Suma el mes actual y el siguiente
        if series[i] > 0 and series[i] + series[i+1] >= umbral:
            discretized[i] = 1
            #print(f"Mes {i}: {series[i]} + {series[i+1]} = {series[i] + series[i+1]} >= {umbral}")
    
    # El último punto se compara solo
    if series[-1] >= umbral:
        discretized[-1] = 1
        
    return discretized

def _clean_consecutive_ones(series):
    """
    """
    cleaned = np.copy(series)
    for i in range(len(cleaned) - 1):
        if cleaned[i] == 1 and cleaned[i+1] == 1:
            cleaned[i+1] = 0
    return cleaned

def _calculate_gaps(series,min_picos):
    indices_picos = []
    if np.sum(series == 1) >= min_picos: 
        contador = 0
        for index,i in enumerate(series):
            if i == 1:
                if index !=0:
                    indices_picos.append(contador)
                    contador = 0
            else:
                contador += 1
            
            if index == len(series)-1 and i== 0:
                indices_picos.append(contador)
    
    return indices_picos

def _detectar_estacionalidad(
    df_ventas: pd.DataFrame,
    historico_ventas: int = 12,
    today: pd.Timestamp = pd.Timestamp.today().normalize(),
    min_picos: int = 2
):
    """
    Detecta si una subcategoría tiene un patrón de compra estacional.
    Devuelve un puntaje de estacionalidad (0 a 1) y los gaps entre picos.
    """
    # 1. Crear la ventana de historico_vetas
    today = pd.to_datetime(today).normalize()
    # 13 meses hacia atrás desde el inicio del mes actual
    start_date_hist = (today.replace(day=1) - pd.DateOffset(months=historico_ventas))
    end_date_hist = today
    full_date_range = pd.date_range(start=start_date_hist, end=end_date_hist, freq='MS')
    df_ventas = df_ventas[
        (df_ventas["DIM_PERIODO"] >= start_date_hist) &
        (df_ventas["DIM_PERIODO"] <= end_date_hist)
    ]

    df_ventas["mes_anio"] = df_ventas["DIM_PERIODO"].dt.strftime("%Y-%m")

    df_vetas_g = df_ventas.groupby(["COD_SUBCATEGORIA","mes_anio"]).agg(
        ventas=("DIM_FACTURA", "count")
    ).reset_index()

    #build the serie 
    df_date_full = pd.DataFrame(
        {
            "mes_anio": full_date_range.strftime("%Y-%m"),
            "key": 1
        }
    )
    df_categorias_f = pd.DataFrame(df_vetas_g["COD_SUBCATEGORIA"].unique(), columns=["COD_SUBCATEGORIA"])
    df_categorias_f["key"] = 1
    df_date_full = df_date_full.merge(df_categorias_f, on="key", how="outer").drop("key", axis=1)
    
    df_date_full = df_date_full.merge(df_vetas_g, on=["COD_SUBCATEGORIA", "mes_anio"], how="left")
    df_date_full["ventas"] = df_date_full["ventas"].fillna(0)
    df_date_full = df_date_full[["COD_SUBCATEGORIA", "mes_anio", "ventas"]]

    # El umbral es la media de transacciones de los meses que SÍ tuvieron compras
    df_agg = df_date_full.groupby("COD_SUBCATEGORIA").agg(
        umbral=("ventas", lambda x: math.floor(x[x>0].mean()) if (x>0).any() else 0),
        serie = ("ventas",lambda x: x.tolist())
    ).reset_index()
    
    df_agg["serie_binaria"] = df_agg.apply(lambda x: _discretize_series(x["serie"], x["umbral"]), axis=1)
    df_agg["serie_limpia"] = df_agg.apply(lambda x: _clean_consecutive_ones(x["serie_binaria"]), axis=1)
    
    # 5. Calcular gaps y puntaje de estacionalidad
    df_agg["indices_picos"] = df_agg.apply(lambda x: _calculate_gaps(x["serie_limpia"],min_picos), axis=1)

    df_agg['CV'] = df_agg.apply(lambda x: np.std(x['indices_picos']) / np.mean(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)
    df_agg["mean_indices_picos"] = df_agg.apply(lambda x: np.mean(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)
    df_agg["std_indices_picos"] = df_agg.apply(lambda x: np.std(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)

    a = 7.85  # pendiente
    b = 0.6   # CV donde score = 0.5
    a = -1.85
    b = 2.5

    df_agg['puntaje'] = df_agg.apply(lambda x: np.exp(a * np.power(x['CV'],b)), axis=1)


    return df_agg[["COD_SUBCATEGORIA", "puntaje","umbral","serie","serie_binaria","serie_limpia","indices_picos","CV","mean_indices_picos","std_indices_picos"]]


def compute_seasonality_features(df_family: pd.DataFrame, fecha_corte: pd.Timestamp, months: int =3) -> pd.DataFrame:
    # 1) Ventanas
    inicio_actual = fecha_corte - pd.DateOffset(months=months)
    mask_actual = (df_family["DIM_PERIODO"] > inicio_actual) & (df_family["DIM_PERIODO"] <= fecha_corte)

    inicio_pasado = inicio_actual - pd.DateOffset(months=12)
    fin_pasado = fecha_corte - pd.DateOffset(months=12)
    mask_pasado = (df_family["DIM_PERIODO"] > inicio_pasado) & (df_family["DIM_PERIODO"] <= fin_pasado)

    # 2) Compras por trimestre
    actuales = (
        df_family[mask_actual]
        .groupby("COD_SUBCATEGORIA")["DIM_FACTURA"]
        .nunique()
        .rename("compras_trim_actual")
    )

    pasados = (
        df_family[mask_pasado]
        .groupby("COD_SUBCATEGORIA")["DIM_FACTURA"]
        .nunique()
        .rename("compras_trim_pasado")
    )

    # 3) Base
    base = pd.DataFrame({"COD_SUBCATEGORIA": df_family["COD_SUBCATEGORIA"].unique()})
    base = base.merge(actuales, on="COD_SUBCATEGORIA", how="left")
    base = base.merge(pasados, on="COD_SUBCATEGORIA", how="left")
    base[["compras_trim_actual", "compras_trim_pasado"]] = base[
        ["compras_trim_actual", "compras_trim_pasado"]
    ].fillna(0)


    A = base["compras_trim_actual"].astype(float)
    P = base["compras_trim_pasado"].astype(float)

    # Caso general P > 0
    need_raw = pd.Series(0.0, index=base.index)

    mask_P_pos = P > 0
    need_raw.loc[mask_P_pos] = (P[mask_P_pos] - A[mask_P_pos]) / P[mask_P_pos]

    # Recorte a [0,1]
    need = need_raw.clip(lower=0.0, upper=1.0)

    # Caso P == 0
    mask_P_zero = (P == 0)

    # Subcasos:
    # - P=0, A=0  -> no hay patrón ni compra: 0
    # - P=0, A>0  -> compró este año, sin patrón histórico del trimestre: también 0 (no backlog)
    #   SE PORDIA  poner algo mínimo tipo 0.1
    need.loc[mask_P_zero] = 0.0

    base["season_ratio"] = need

    #estacioonalidad
    estacionalidad = _detectar_estacionalidad(df_ventas=df_family.copy(),historico_ventas=12,today=fecha_corte,min_picos=2)


    base = base.merge(estacionalidad, on="COD_SUBCATEGORIA", how="left")

    factor = 0.5 + 0.5 * base["puntaje"]
    base["season_ratio"] = base["season_ratio"] * factor

    #["COD_SUBCATEGORIA", "puntaje","umbral","serie","serie_binaria","serie_limpia","indices_picos","CV"]
    
    return base[["COD_SUBCATEGORIA", "season_ratio","compras_trim_actual", "compras_trim_pasado","puntaje","umbral","serie","serie_binaria","serie_limpia","indices_picos","CV","mean_indices_picos","std_indices_picos"]]


def compute_features_for_family(
    df_family: pd.DataFrame,   # es el dataset de ventas filtrado por la familia sin group by de nada
    family_code: int
) -> pd.DataFrame:
    df_family = df_family[df_family["DIM_PERIODO"] < CUTOFF_DATE].copy()
    if df_family.empty:
        return pd.DataFrame()

    fecha_corte = CUTOFF_DATE

    subcat_agg = (
        df_family.groupby("COD_SUBCATEGORIA")
        .agg(
            total_cantidad=("CANTIDAD_SUELTA", "sum"),
            total_venta_neta=("VENTA_NETA", "sum"),
            total_descuento=("DESCUENTO", "sum"),
            total_pvp=("PVP", "sum"),
            promedio_pvp=("PVP", "mean"),
            facturas_unicas=("DIM_FACTURA", "nunique"),
            registros=("DIM_FACTURA", "count"),
            primera_compra=("DIM_PERIODO", "min"),
            ultima_compra=("DIM_PERIODO", "max"),
        )
        .reset_index()
    )

    if subcat_agg.empty:
        return pd.DataFrame()

    subcat_agg = subcat_agg.sort_values("total_venta_neta", ascending=False).reset_index(drop=True)
    #columnas para tracking compleot ciclos de castigo 
    
    columnas_ciclos_estacionales = ["CODIGO_FAMILIA", "COD_SUBCATEGORIA", "ciclo_dias", "fuerza_castigo", "gaps_originales_dias", "gaps_suavizados_dias"]
    ciclos_estacionales = calcular_ciclos_por_bloques(
        df_ventas=df_family, #es el raw dataset vetnas filtrado solo por la famialia
        familia_id=family_code,  #coidigo de familia para debug mas que todo
        meses_historico=12, #meses historicos para calcular los ciclos hacia a tras desde fecha de corte
        periodo_dias =7,#aplanamos datos independientemente del numero de compras por este perido cuenta como uno
        min_bloques_compra=3, #es el minimo de compras que tiene que tener para poder caclcular el ciclo
        max_compras_recientes = 10, # es el maximo numero de compras spara calcular el ciclo si tiene mas se acota segun su periodo
        today=fecha_corte)

    columnas_recencia = ["COD_SUBCATEGORIA", "recencia_hl","castigo_recencia","l_compra_sobre_ciclo","dias_desde_ultima_compra","recencia"]
    recency_features =  compute_recency_features(subcat_agg=subcat_agg, #mandado mi data agrupada ya 
                                              ciclos_estacionales=ciclos_estacionales, 
                                              fecha_corte=fecha_corte)

    columnas_freq = ["COD_SUBCATEGORIA", "freq_score","avg_compras","compras","ratio"]
    freq_features = compute_frequency_features(df_family=df_family,
                                            fecha_corte=fecha_corte,
                                            ciclos_estacionales=ciclos_estacionales,
                                            periodo_dias=7) #este perido no esta haciendo nada
    features_final = recency_features.merge(freq_features, on="COD_SUBCATEGORIA", how="left")

    
    columnas_sow = ["COD_SUBCATEGORIA", "sow_24m","transacciones_netas"]
    sow_features = compute_sow_features(df_family=df_family, fecha_corte=fecha_corte)
    features_final = features_final.merge(sow_features, on="COD_SUBCATEGORIA", how="left")

    columnas_seasonality = ["COD_SUBCATEGORIA", "puntaje","umbral","serie","serie_binaria","serie_limpia","indices_picos","CV","mean_indices_picos","std_indices_picos"]
    seasonality_features = compute_seasonality_features(df_family=df_family, fecha_corte=fecha_corte)
    features_final = features_final.merge(seasonality_features, on="COD_SUBCATEGORIA", how="left")

    features_final = features_final.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.fillna(0.0)

    features_final["score_final"] = (
        0.4 * features_final["recencia_hl"]
        + 0.3 * features_final["freq_score"]
        + 0.1 * features_final["sow_24m"]
        + 0.2 * features_final["season_ratio"]
    )

    score_columns = [
        "COD_SUBCATEGORIA",
        "recencia_hl",
        "freq_score",
        "sow_24m",
        "season_ratio",
        "score_final",
    ]

    # --- Renombrado de columnas por origen ---
    rename_map = {}

    def add_renames(cols, prefix):
        for col in cols:
            if col in features_final.columns and col not in score_columns:
                rename_map[col] = f"{prefix}_{col}"

    add_renames(columnas_ciclos_estacionales, "Ciclos")
    add_renames(columnas_recencia, "Recencia")
    add_renames(columnas_freq, "Freq")
    add_renames(columnas_sow, "Sow")
    add_renames(columnas_seasonality, "Seasonality")

    features_final = features_final.rename(columns=rename_map)

    # Construir lista final de columnas: score_columns + las renombradas
    final_cols = list(score_columns) + list(rename_map.values())
    # Eliminar duplicados manteniendo orden
    final_cols = list(dict.fromkeys(final_cols))
    # Filtrar solo existentes
    final_cols = [c for c in final_cols if c in features_final.columns]

    features_final = features_final[final_cols]
    features_final["nucleo"] = family_code

    return features_final

def process_all_families(
    df_raw: pd.DataFrame,
    families: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if families is None:
        families_iter = sorted(df_raw["CODIGO_FAMILIA"].unique())
    else:
        family_set = {int(fam) for fam in families}
        families_iter = [fam for fam in sorted(df_raw["CODIGO_FAMILIA"].unique()) if fam in family_set]

    results: List[pd.DataFrame] = []
    for family_code in families_iter:
        df_family = df_raw[df_raw["CODIGO_FAMILIA"] == family_code].copy()
        features_family = compute_features_for_family(df_family, family_code)
        if features_family.empty:
            continue
        results.append(features_family)
        print(f"[FeatureEngineering] Núcleo {family_code}: {len(features_family)} subcategorías procesadas")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def main() -> None:
    print("Cargando dataset histórico...")
    df_raw = load_historical_dataset(HISTORICAL_FILE)
    print(f"Registros totales: {len(df_raw)} | Núcleos únicos: {df_raw['CODIGO_FAMILIA'].nunique()}")


    features_all = process_all_families(df_raw)
    if features_all.empty:
        print("No se generaron features para ningún núcleo.")
        return

    features_all.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo consolidado guardado en: {OUTPUT_FILE}")

    return features_all,df_raw


if __name__ == "__main__":
    features_all,df_raw = main()
    print("finalizado proceso")
    
    # %%
    # Post-procesamiento - Comentado para permitir importación limpia del módulo
    """
    columnas_subcategrio = ["COD_SUBCATEGORIA","NOMBRE_SUBCATEGORIA"]
    df_raw_subcat = df_raw[columnas_subcategrio].drop_duplicates()
    
    features_all_subcat = features_all.merge(df_raw_subcat, on="COD_SUBCATEGORIA", how="left")
    columnas_finales = [
    'nucleo',                        # CODIGO FAMILIA
    'COD_SUBCATEGORIA',              # COD SUBCATEGORIA
    'NOMBRE_SUBCATEGORIA',           # NOMBRE SUBCATEGORIA
    'recencia_hl',                   # RECENCIA
    'freq_score',                    # FRECUENCIA
    'sow_24m',                       # SOW
    'season_ratio',                  # ESTACIONALIDAD
    'score_final',                   # SCORE SUBCATEGORIA
    'Ciclos_ciclo_dias',             # CICLO
    'Recencia_dias_desde_ultima_compra', # DIAS ULTIMA COMPRA
    'Seasonality_mean_indices_picos',# MEDIA ESTACION
    'Seasonality_std_indices_picos'  # STD DEV ESTACION
]
features_all_subcat = features_all_subcat[columnas_finales]

features_all_subcat = features_all_subcat.sort_values(by="score_final", ascending=False).reset_index(drop=True).reset_index()
#save csv 
#remane index to top
#reamne columns 
'''
    'nucleo',                        # CODIGO FAMILIA
    'COD_SUBCATEGORIA',              # COD SUBCATEGORIA
    'NOMBRE_SUBCATEGORIA',           # NOMBRE SUBCATEGORIA
    'recencia_hl',                   # RECENCIA
    'freq_score',                    # FRECUENCIA
    'sow_24m',                       # SOW
    'season_ratio',                  # ESTACIONALIDAD
    'score_final',                   # SCORE SUBCATEGORIA
    'Ciclos_ciclo_dias',             # CICLO
    'Recencia_dias_desde_ultima_compra', # DIAS ULTIMA COMPRA
    'Seasonality_mean_indices_picos',# MEDIA ESTACION
    'Seasonality_std_indices_picos'  # STD DEV ESTACION
'''
features_all_subcat.rename(columns={"index": "top",
"nucleo":"CODIGO_FAMILIA",
"COD_SUBCATEGORIA":"COD_SUBCATEGORIA",
"NOMBRE_SUBCATEGORIA":"NOMBRE_SUBCATEGORIA",
"recencia_hl":"RECENCIA",
"freq_score":"FRECUENCIA",
"sow_24m":"SOW",
"season_ratio":"ESTACIONALIDAD",
"score_final":"SCORE_SUBCATEGORIA",
"Ciclos_ciclo_dias":"CICLO",
"Recencia_dias_desde_ultima_compra":"DIAS_ULTIMA_COMPRA",
"Seasonality_mean_indices_picos":"MEDIA_ESTACION",
"Seasonality_std_indices_picos":"STD_DEV_ESTACION"}, inplace=True)



features_all_subcat.to_csv("features_with_subcat_names.csv", index=False)
#save in excel format
features_all_subcat.to_excel("features_with_subcat_names.xlsx", index=False)

#%%

#read item.xlsx
item = pd.read_excel("item.xlsx")
# %%
nombre_divicion = ["FARMA","CONSUMO"]
item = item[item['NOMBRE_DIVISION_COMERCIAL'].isin(nombre_divicion)]
#DROP DUPLICADOS 
colums_unicas = ['COD_ITEM', 'COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL']
item = item.drop_duplicates(subset=colums_unicas).reset_index(drop=True)

columnas_uso = ['COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL','NOMBRE_DIVISION_COMERCIAL']
item_merge = item[columnas_uso].drop_duplicates(subset=['COD_SUBCATEGORIA']).copy()
features_all_subcat = features_all_subcat.merge(item_merge, on="COD_SUBCATEGORIA", how="left")

# --- FARMA ---
feature_farma = features_all_subcat[features_all_subcat['NOMBRE_DIVISION_COMERCIAL'] == 'FARMA'].copy()
# Ordenar por nucleo y score para ranking correcto por familia
feature_farma = feature_farma.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA"], ascending=[True, False])
# Ranking por nucleo
feature_farma['top_s'] = feature_farma.groupby("CODIGO_FAMILIA").cumcount() + 1

# --- CONSUMO ---
feature_consumo = features_all_subcat[features_all_subcat['NOMBRE_DIVISION_COMERCIAL'] == 'CONSUMO'].copy()
# Ordenar por nucleo y score para ranking correcto por familia
feature_consumo = feature_consumo.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA"], ascending=[True, False])
# Ranking por nucleo
feature_consumo['top_s'] = feature_consumo.groupby("CODIGO_FAMILIA").cumcount() + 1

    feature_concat = pd.concat([feature_farma, feature_consumo], ignore_index=True)
    features_all_subcat = feature_concat.copy()
    features_all_subcat.to_excel("features_with_subcat_names.xlsx", index=False)
    """
