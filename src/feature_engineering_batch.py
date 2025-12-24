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
        encoding='utf-8',
        sep=';')
        

    # Convertir fechas
    df["DIM_PERIODO"] = pd.to_datetime(
        df["DIM_PERIODO"],
        format="%d-%b-%y", # ejemplo: 24-Nov-25
        errors="coerce",
    )

    # Convertir columnas numéricas
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # Convertir columnas int con coercion (rellenar NaN antes de convertir)
    for col in RAW_DTYPES.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)

    #filtrar solo familia especifica 1719363028
    #ceduleas  = [1102088158,1719363028]
    #df = df[df["CODIGO_FAMILIA"].isin(ceduleas)]


    return df

def calcular_cv_normalizado(gaps_dias):
    """
    Normaliza gaps dividiendo por el mínimo y calcula el Coeficiente de Variación (CV).
    
    Returns:
        cv (float): Coeficiente de variación normalizado
        gaps_norm (array): Gaps normalizados
    """
    arr = np.asarray(gaps_dias, dtype=float)
    
    if arr.size < 2:
        return 999.0, arr  # CV infinito = muy irregular
    
    # Normalización: dividir por el mínimo
    min_gap = np.min(arr)
    if min_gap == 0:
        return 999.0, arr
    
    gaps_norm = arr / min_gap
    
    # Calcular CV normalizado
    mean_norm = np.mean(gaps_norm)
    std_norm = np.std(gaps_norm)
    
    cv = std_norm / mean_norm if mean_norm > 0 else 999.0
    
    return cv, gaps_norm

def calcular_ciclos_cortos(
    df_ventas,
    familia_id,
    subcat,
    meses_historico=12,
    periodo_dias=7,
    min_compras=3,
    max_compras_recientes=10,
    cv_threshold=0.5,
    today=pd.Timestamp.today()
):
    """
    Detecta ciclos cortos/medios (hasta ~6 meses).
    Usa ventana de 12 meses y bloques de 7 días.
    """
    today = today.normalize()
    fecha_inicio = today - pd.DateOffset(months=meses_historico)
    
    df_sub = df_ventas[
        (df_ventas["CODIGO_FAMILIA"] == familia_id) &
        (df_ventas["COD_SUBCATEGORIA"] == subcat) &
        (pd.to_datetime(df_ventas["DIM_PERIODO"]) >= fecha_inicio)
    ].copy()
    
    if df_sub.empty:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "sin_datos"}
    
    # Calcular bloques
    dias_desde_inicio = (df_sub["DIM_PERIODO"] - fecha_inicio).dt.days
    df_sub["bloque"] = dias_desde_inicio // periodo_dias
    
    # Ordenar por fecha y tomar bloques únicos
    df_sub = df_sub.sort_values("DIM_PERIODO", ascending=False)
    bloques_con_compra = np.sort(df_sub["bloque"].unique())
    
    # Limitar a últimas N compras
    if len(bloques_con_compra) > max_compras_recientes:
        bloques_con_compra = bloques_con_compra[-max_compras_recientes:]
    
    # Verificar mínimo de compras
    if len(bloques_con_compra) < min_compras:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "pocas_compras_corto"}
    
    # Calcular gaps de BLOQUES (para CV - suavizado)
    gaps_bloques = np.diff(bloques_con_compra)
    if len(gaps_bloques) == 0:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "sin_gaps"}
    
    gaps_dias_bloques = gaps_bloques * periodo_dias
    
    # Calcular CV normalizado usando gaps de bloques (suavizados)
    cv, gaps_norm = calcular_cv_normalizado(gaps_dias_bloques)
    
    # Calcular gaps REALES (días exactos entre fechas de compra)
    # Para esto, tomamos las fechas únicas ordenadas
    fechas_unicas = df_sub.groupby('bloque')['DIM_PERIODO'].max().sort_values()
    if len(bloques_con_compra) > max_compras_recientes:
        fechas_unicas = fechas_unicas.iloc[-max_compras_recientes:]
    
    gaps_dias_reales = np.diff(fechas_unicas).astype('timedelta64[D]').astype(int)
    gaps_dias_reales = gaps_dias_reales.tolist()
    
    # Decidir si es cíclico
    if cv <= cv_threshold:
        # Ciclo promedio basado en gaps REALES
        ciclo_dias = float(np.mean(gaps_dias_reales))
        
        # Validar coherencia: cortos deben ser < 180 días (6 meses)
        if ciclo_dias >= 180:
            return {"ciclo_dias": 0, "cv": cv, "tipo": "no_ciclico", "razon": "ciclo_muy_largo_para_corto"}
        
        return {
            "ciclo_dias": ciclo_dias,
            "cv": cv,
            "tipo": "corto",
            "gaps_originales": gaps_dias_reales,  # Días REALES
            "gaps_normalizados": gaps_norm.tolist()
        }
    else:
        return {"ciclo_dias": 0, "cv": cv, "tipo": "no_ciclico", "razon": "cv_alto_corto"}


def calcular_ciclos_largos(
    df_ventas,
    familia_id,
    subcat,
    meses_historico=36,
    periodo_dias=30,
    min_compras=4,
    cv_threshold=0.7,
    today=pd.Timestamp.today()
):
    """
    Detecta ciclos largos (6 meses a 2 años).
    Usa ventana de 36 meses y bloques de 30 días.
    """
    today = today.normalize()
    fecha_inicio = today - pd.DateOffset(months=meses_historico)
    
    df_sub = df_ventas[
        (df_ventas["CODIGO_FAMILIA"] == familia_id) &
        (df_ventas["COD_SUBCATEGORIA"] == subcat) &
        (pd.to_datetime(df_ventas["DIM_PERIODO"]) >= fecha_inicio)
    ].copy()
    
    if df_sub.empty:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "sin_datos_largo"}
    
    # Calcular bloques
    dias_desde_inicio = (df_sub["DIM_PERIODO"] - fecha_inicio).dt.days
    df_sub["bloque"] = dias_desde_inicio // periodo_dias
    
    # Bloques únicos ordenados
    bloques_con_compra = np.sort(df_sub["bloque"].unique())
    
    # Verificar mínimo de compras
    if len(bloques_con_compra) < min_compras:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "pocas_compras_largo"}
    
    # Calcular gaps de BLOQUES (para CV - suavizado)
    gaps_bloques = np.diff(bloques_con_compra)
    if len(gaps_bloques) == 0:
        return {"ciclo_dias": 0, "cv": 999, "tipo": "no_ciclico", "razon": "sin_gaps_largo"}
    
    gaps_dias_bloques = gaps_bloques * periodo_dias
    
    # Calcular CV normalizado usando gaps de bloques (suavizados)
    cv, gaps_norm = calcular_cv_normalizado(gaps_dias_bloques)
    
    # Calcular gaps REALES (días exactos entre fechas de compra)
    fechas_unicas = df_sub.groupby('bloque')['DIM_PERIODO'].max().sort_values()
    gaps_dias_reales = np.diff(fechas_unicas).astype('timedelta64[D]').astype(int)
    gaps_dias_reales = gaps_dias_reales.tolist()
    
    # Decidir si es cíclico
    if cv <= cv_threshold:
        # Ciclo promedio basado en gaps REALES
        ciclo_dias = float(np.mean(gaps_dias_reales))
        
        # Validar coherencia: largos deben ser >= 180 días (6 meses)
        if ciclo_dias < 180:
            return {"ciclo_dias": 0, "cv": cv, "tipo": "no_ciclico", "razon": "ciclo_muy_corto_para_largo"}
        
        return {
            "ciclo_dias": ciclo_dias,
            "cv": cv,
            "tipo": "largo",
            "gaps_originales": gaps_dias_reales,  # Días REALES
            "gaps_normalizados": gaps_norm.tolist()
        }
    else:
        return {"ciclo_dias": 0, "cv": cv, "tipo": "no_ciclico", "razon": "cv_alto_largo"}


def calcular_ciclos_por_bloques(
    df_ventas,
    familia_id,
    today=pd.Timestamp.today()
):
    """
    Orquestador: intenta primero ciclos cortos, luego largos.
    Retorna DataFrame con resultados para todas las subcategorías.
    """
    today = today.normalize()
    
    # Obtener subcategorías de la familia
    df_fam = df_ventas[df_ventas["CODIGO_FAMILIA"] == familia_id].copy()
    if df_fam.empty:
        return pd.DataFrame()
    
    subcategorias = df_fam["COD_SUBCATEGORIA"].unique()
    resultados = []
    
    for subcat in subcategorias:
        # FASE 1: Intentar ciclos cortos
        resultado_corto = calcular_ciclos_cortos(
            df_ventas=df_ventas,
            familia_id=familia_id,
            subcat=subcat,
            today=today
        )
        
        if resultado_corto["ciclo_dias"] > 0:
            # Encontró ciclo corto
            resultados.append({
                "CODIGO_FAMILIA": familia_id,
                "COD_SUBCATEGORIA": subcat,
                "ciclo_dias": resultado_corto["ciclo_dias"],
                "cv": resultado_corto["cv"],
                "tipo_ciclo": resultado_corto["tipo"],
                "gaps_originales_dias": resultado_corto.get("gaps_originales", []),
                "gaps_normalizados": resultado_corto.get("gaps_normalizados", [])
            })
        else:
            # FASE 2: Intentar ciclos largos
            resultado_largo = calcular_ciclos_largos(
                df_ventas=df_ventas,
                familia_id=familia_id,
                subcat=subcat,
                today=today
            )
            
            resultados.append({
                "CODIGO_FAMILIA": familia_id,
                "COD_SUBCATEGORIA": subcat,
                "ciclo_dias": resultado_largo["ciclo_dias"],
                "cv": resultado_largo["cv"],
                "tipo_ciclo": resultado_largo["tipo"],
                "gaps_originales_dias": resultado_largo.get("gaps_originales", []),
                "gaps_normalizados": resultado_largo.get("gaps_normalizados", [])
            })
    
    df_resultado = pd.DataFrame(resultados)
    
    # Ordenar: cíclicos primero, por CV ascendente
    if not df_resultado.empty:
        df_resultado = df_resultado.sort_values(["ciclo_dias", "cv"], ascending=[False, True]).reset_index(drop=True)
    
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


def compute_frequency_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula features de frecuencia con ventanas adaptativas según tipo de ciclo.
    - Ciclos cortos: ventana de 180 días (6 meses)
    - Ciclos largos: ventana de max(360, ciclo*3) días
    """
    resultados = []

    
    for _, row in ciclos_estacionales.iterrows():
        subcat = row["COD_SUBCATEGORIA"]
        ciclo_dias = row["ciclo_dias"]
        tipo_ciclo = row.get("tipo_ciclo", "no_ciclico")
        
        # Determinar ventana según tipo de ciclo
        if tipo_ciclo == "largo":
            # Ciclos largos: mínimo 1 año o 3 ciclos
            ventana_dias = max(360, int(ciclo_dias * 3))
        else:
            # Ciclos cortos o no cíclicos: 180 días (6 meses)
            ventana_dias = FREQUENCY_WINDOW_DAYS
        
        ventana_inicio = fecha_corte - pd.Timedelta(days=ventana_dias)
        recientes = df_family[
            (df_family["COD_SUBCATEGORIA"] == subcat) &
            (df_family["DIM_PERIODO"] >= ventana_inicio)
        ].copy()
        
        compras_reales = recientes["DIM_FACTURA"].count()
        
        # Calcular compras esperadas
        if ciclo_dias > 0:
            avg_compras = (ventana_dias / ciclo_dias) * 1.2
        else:
            avg_compras = 0.0

        # Calcular score según ratio
        if avg_compras > 0:
            ratio = compras_reales / avg_compras
            
            # Curva de frecuencia
            alpha = 0.25
            base_min = 0.4
            base_max = 1.0
            r_max = 1.6
            
            if ratio <= 1:
                raw_low = (1 - ratio) ** alpha
                freq_score = base_min + (base_max - base_min) * raw_low
            elif ratio <= r_max:
                exceso = ratio - 1
                freq_score = base_min * max(0, 1 - exceso / (r_max - 1))
            else:
                freq_score = 0.0
        else:
            ratio = 10.0
            freq_score = 0.0
        
        resultados.append({
            "COD_SUBCATEGORIA": subcat,
            "freq_score": freq_score,
            "avg_compras": avg_compras,
            "compras": compras_reales,
            "ratio": ratio,
            "ventana_dias": ventana_dias
        })
    
    return pd.DataFrame(resultados)


def compute_sow_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula Share of Wallet con pesos adaptativos según tipo de ciclo.
    - Ciclos cortos: peso_12m=7, peso_24m=3 (sesgo reciente)
    - Ciclos largos: peso_12m=1, peso_24m=1 (sin sesgo)
    """
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
    
    # Crear mapa de tipo de ciclo
    tipo_ciclo_map = ciclos_estacionales.set_index("COD_SUBCATEGORIA")["tipo_ciclo"].to_dict()
    
    # Aplicar pesos según tipo de ciclo
    def aplicar_pesos(row, periodo):
        subcat = row["COD_SUBCATEGORIA"]
        tipo = tipo_ciclo_map.get(subcat, "no_ciclico")
        
        if tipo == "largo":
            # Pesos iguales para ciclos largos
            peso = 1
        else:
            # Sesgo reciente para ciclos cortos
            peso = 7 if periodo == "12m" else 3
        
        return row["transacciones_netas"] * peso
    
    sow_agg_12m["transacciones_netas"] = sow_agg_12m.apply(lambda r: aplicar_pesos(r, "12m"), axis=1)
    sow_agg_24m["transacciones_netas"] = sow_agg_24m.apply(lambda r: aplicar_pesos(r, "24m"), axis=1)

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


def compute_seasonality_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula features de estacionalidad con ventanas adaptativas.
    - Ciclos cortos: 3 meses actual vs 3 meses hace 1 año
    - Ciclos largos: 6 meses actual vs promedio de (hace 1, 2 y 3 años)
    """
    tipo_ciclo_map = ciclos_estacionales.set_index("COD_SUBCATEGORIA")["tipo_ciclo"].to_dict()
    resultados = []

    
    for subcat in df_family["COD_SUBCATEGORIA"].unique():
        tipo = tipo_ciclo_map.get(subcat, "no_ciclico")
        
        if tipo == "largo":
            # Ciclos largos: 6 meses actual vs promedio de 3 años
            months_actual = 6
            
            # Ventana actual
            inicio_actual = fecha_corte - pd.DateOffset(months=months_actual)
            mask_actual = (
                (df_family["COD_SUBCATEGORIA"] == subcat) &
                (df_family["DIM_PERIODO"] > inicio_actual) & 
                (df_family["DIM_PERIODO"] <= fecha_corte)
            )
            
            # Ventanas pasadas: hace 1, 2 y 3 años
            compras_pasadas = []
            for years_ago in [1, 2, 3]:
                inicio = fecha_corte - pd.DateOffset(months=months_actual + 12*years_ago)
                fin = fecha_corte - pd.DateOffset(months=12*years_ago)
                mask = (
                    (df_family["COD_SUBCATEGORIA"] == subcat) &
                    (df_family["DIM_PERIODO"] > inicio) & 
                    (df_family["DIM_PERIODO"] <= fin)
                )
                compras = df_family[mask]["DIM_FACTURA"].nunique()
                compras_pasadas.append(compras)
            
            compras_actual = df_family[mask_actual]["DIM_FACTURA"].nunique()
            compras_pasado_promedio = np.mean(compras_pasadas)
            
        else:
            # Ciclos cortos: 3 meses actual vs 3 meses hace 1 año
            months_actual = 3
            
            inicio_actual = fecha_corte - pd.DateOffset(months=months_actual)
            mask_actual = (
                (df_family["COD_SUBCATEGORIA"] == subcat) &
                (df_family["DIM_PERIODO"] > inicio_actual) & 
                (df_family["DIM_PERIODO"] <= fecha_corte)
            )
            
            inicio_pasado = inicio_actual - pd.DateOffset(months=12)
            fin_pasado = fecha_corte - pd.DateOffset(months=12)
            mask_pasado = (
                (df_family["COD_SUBCATEGORIA"] == subcat) &
                (df_family["DIM_PERIODO"] > inicio_pasado) & 
                (df_family["DIM_PERIODO"] <= fin_pasado)
            )
            
            compras_actual = df_family[mask_actual]["DIM_FACTURA"].nunique()
            compras_pasado_promedio = df_family[mask_pasado]["DIM_FACTURA"].nunique()
        
        # Calcular need
        if compras_pasado_promedio > 0:
            need = (compras_pasado_promedio - compras_actual) / compras_pasado_promedio
            need = max(0.0, min(1.0, need))
        else:
            need = 0.0
        
        resultados.append({
            "COD_SUBCATEGORIA": subcat,
            "season_ratio_base": need,
            "compras_trim_actual": compras_actual,
            "compras_trim_pasado": compras_pasado_promedio
        })
    
    base = pd.DataFrame(resultados)
    
    # Detección de estacionalidad (usa 12 meses)
    estacionalidad = _detectar_estacionalidad(
        df_ventas=df_family.copy(),
        historico_ventas=12,
        today=fecha_corte,
        min_picos=2
    )
    
    base = base.merge(estacionalidad, on="COD_SUBCATEGORIA", how="left")
    
    # Aplicar factor de estacionalidad
    factor = 0.5 + 0.5 * base["puntaje"].fillna(0)
    base["season_ratio"] = base["season_ratio_base"] * factor
    
    return base[["COD_SUBCATEGORIA", "season_ratio", "compras_trim_actual", "compras_trim_pasado", 
                 "puntaje", "umbral", "serie", "serie_binaria", "serie_limpia", "indices_picos", 
                 "CV", "mean_indices_picos", "std_indices_picos"]]


def compute_features_for_family(
    df_family: pd.DataFrame,   # es el dataset de ventas filtrado por la familia sin group by de nada
    family_code: int,
    fecha_corte: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Calcula features para una familia.
    
    Args:
        df_family: DataFrame con ventas de la familia
        family_code: Código de la familia
        fecha_corte: Fecha de corte para cálculo de features (default: CUTOFF_DATE global)
    """
    if fecha_corte is None:
        fecha_corte = CUTOFF_DATE
    
    df_family = df_family[df_family["DIM_PERIODO"] < fecha_corte].copy()
    if df_family.empty:
        return pd.DataFrame()

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
    
    # Calcular ciclos (ahora con sistema de 2 fases: cortos y largos)
    ciclos_estacionales = calcular_ciclos_por_bloques(
        df_ventas=df_family,
        familia_id=family_code,
        today=fecha_corte
    )
    
    if ciclos_estacionales.empty:
        return pd.DataFrame()

    # Calcular features de recencia
    recency_features = compute_recency_features(
        subcat_agg=subcat_agg,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    
    # Calcular features de frecuencia (con ventanas adaptativas)
    freq_features = compute_frequency_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    features_final = recency_features.merge(freq_features, on="COD_SUBCATEGORIA", how="left")
    
    # Calcular SOW (con pesos adaptativos)
    sow_features = compute_sow_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    features_final = features_final.merge(sow_features, on="COD_SUBCATEGORIA", how="left")
    
    # Calcular estacionalidad (con ventanas adaptativas)
    seasonality_features = compute_seasonality_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
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
    
    # Definir columnas por origen
    columnas_ciclos = ["CODIGO_FAMILIA", "COD_SUBCATEGORIA", "ciclo_dias", "cv", "tipo_ciclo", "gaps_originales_dias", "gaps_normalizados"]
    columnas_recencia = ["COD_SUBCATEGORIA", "recencia_hl", "castigo_recencia", "l_compra_sobre_ciclo", "dias_desde_ultima_compra", "recencia"]
    columnas_freq = ["COD_SUBCATEGORIA", "freq_score", "avg_compras", "compras", "ratio", "ventana_dias"]
    columnas_sow = ["COD_SUBCATEGORIA", "sow_24m", "transacciones_netas"]
    columnas_seasonality = ["COD_SUBCATEGORIA", "puntaje", "umbral", "serie", "serie_binaria", "serie_limpia", "indices_picos", "CV", "mean_indices_picos", "std_indices_picos"]

    add_renames(columnas_ciclos, "Ciclos")
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
