# ============================================================================
# IMPORTS Y CONFIGURACIÓN
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from feature_engineering_batch import compute_features_for_family, load_historical_dataset

# ============================================================================
# CARGA DE DATOS
# ============================================================================
# Datos históricos de ventas completos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Data"
HISTORICAL_FILE = DATA_DIR / "Historico_08122025.csv"
DATA_RECOMENDATION = BASE_DIR / "Resultados" / "predictions_fnn_final.csv"
SCORE_COL = "fnn_prob"
# ============================================================================
# FUNCIONES
# ============================================================================
#columnas usadas en el histórico
columas_usadas_ventas = ['COD_SUBCATEGORIA', 'NOMBRE_SUBCATEGORIA','COD_ITEM', 'DIM_PERIODO','DIM_FACTURA','CODIGO_FAMILIA']

df_historico = load_historical_dataset(HISTORICAL_FILE)
df_score_subcategoria = pd.read_csv(DATA_RECOMENDATION)
fecha_max_historico = pd.to_datetime('2025-11-30')

def perpare_df_historico(Path: Path, columnas_usadas_ventas: list) -> pd.DataFrame:
    df_historico = load_historical_dataset(Path)
    df_historico_vetas = df_historico[columnas_usadas_ventas].copy()
    df_historico_vetas.rename(columns={'COD_ITEM': 'CODIGO_ITEM'}, inplace=True)

    # Convertir columnas identificadoras a STRING desde el inicio (evita problemas de float)
    columnas_a_string = ['CODIGO_FAMILIA', 'CODIGO_ITEM', 'COD_SUBCATEGORIA', 'COD_LOCAL', 'DIM_FACTURA']
    for col in columnas_a_string:
        if col in df_historico_vetas.columns:
            df_historico_vetas[col] = df_historico_vetas[col].astype(str)
    return df_historico_vetas

def prepare_df_score_subcategoria(df_score_subcategoria: pd.DataFrame) -> pd.DataFrame:
    df_score_subcategoria.rename(columns={'nucleo': 'CODIGO_FAMILIA'}, inplace=True)
    df_score_subcategoria['CODIGO_FAMILIA'] = df_score_subcategoria['CODIGO_FAMILIA'].astype(str)
    df_score_subcategoria['COD_SUBCATEGORIA'] = df_score_subcategoria['COD_SUBCATEGORIA'].astype(str)
    return df_score_subcategoria


def recompra(df_historico: pd.DataFrame,df_score_subcategoria: pd.DataFrame, fecha_max_historico: dt.datetime) -> pd.DataFrame:    
    # Calcular última compra y conteo por ítem-familia
    df_historico_grupby = df_historico.groupby(['CODIGO_ITEM','CODIGO_FAMILIA']).agg(
        last_buy_day=('DIM_PERIODO', 'max'),
        count_buy=('DIM_PERIODO', 'count'),
        penalizacion_120 = ('DIM_PERIODO', lambda x: x[x >= fecha_max_historico - pd.Timedelta(days=180)].count()),
        penalizacion_60 = ('DIM_PERIODO', lambda x: x[x >= fecha_max_historico - pd.Timedelta(days=60)].count()),
        penalizacion_30 = ('DIM_PERIODO', lambda x: x[x >= fecha_max_historico - pd.Timedelta(days=30)].count())
    )


    df_historico_grupby['reapat_count_60'] = df_historico_grupby['penalizacion_60'] - df_historico_grupby['penalizacion_30']
    
    df_historico_grupby['reapat_count_90'] = df_historico_grupby['penalizacion_90'] - df_historico_grupby['penalizacion_60'] - df_historico_grupby['penalizacion_30']
    
    # Calcular días desde última compra
    df_historico_grupby['count_last_day'] = (fecha_max_historico - df_historico_grupby['last_buy_day']).dt.days

    # === 4) Penalización por ciclos (con pisos + tope) ====
    c30 = np.maximum(df_historico_grupby['reapat_count_30'], 0.5)
    c60 = np.maximum(df_historico_grupby['reapat_count_60'], 0.6)
    c90 = np.maximum(df_historico_grupby['reapat_count_90'], 0.4)
    c120 = np.where(df_historico_grupby['penalizacion_120'] > 2, 1, 0.2)
    penalty_raw = c30 * c60 * c90 * c120
    penalty_cap = np.minimum(penalty_raw, 1)

    mu = 30
    s  = 5
    b  = 0.1
    t = df_historico_grupby['count_last_day']
    
    sigmoid = 1 / (1 + np.exp(-(t - mu)/s))
    due = b + (1 - b) * sigmoid

    df_historico_grupby['due_B'] = due

    # === 6) Score final ===
    score_raw = due * penalty_cap
    df_historico_grupby['repeat_final'] = np.minimum(1.0, score_raw)

    return df_historico_grupby
    
def popularidad(df_historico: pd.DataFrame) -> pd.DataFrame:

    t = fecha_max_historico - pd.Timedelta(days=365)

    # Compras del ITEM específico (facturas únicas)
    df_historico_item = df_historico.groupby(
        ['CODIGO_FAMILIA','COD_SUBCATEGORIA','CODIGO_ITEM'].copy()
    ).agg(
        compras_item = ('DIM_FACTURA', 'nunique'),
        compras_last_year = ('DIM_FACTURA', lambda x: x[x >= t].count())

    ).reset_index()

    # Compras TOTALES de la subcategoría (facturas únicas)
    df_subcat_total = df_historico.groupby(
        ['CODIGO_FAMILIA','COD_SUBCATEGORIA'].copy().agg(
            {'DIM_FACTURA': 'nunique'},
            {'CODIGO_ITEM': 'nunique'}
        )
    ).reset_index().rename(columns={'DIM_FACTURA': 'compras_subcat_total', 'CODIGO_ITEM': 'num_items_subcat'})
    
    # Merge back to main dataframe
    df_historico_g = df_subcat_total.merge(df_historico_item, on=['CODIGO_FAMILIA','COD_SUBCATEGORIA'], how='left')

    # Penalización 1: es el único ítem que se compra en esa subcategoría
    pen_compras = np.where(
        df_historico_g['compras_item'] == df_historico_g['compras_subcat_total'],
        0.5,
        1.0
    )

    # --- Compras del ítem en el ÚLTIMO AÑO
    

    # Penalización 2: se ha comprado menos de 2 veces en el último año
    pen_compras_last_year = np.where(
        df_historico_g['compras_last_year'] < 2,
        0.5,
        1.0
    )
    # Pop base (frecuencia relativa en la subcategoría)
    alpha = 1
    df_historico_g['pop_item'] = (
        (df_historico_g['compras_item'] + alpha) /
        (df_historico_g['compras_subcat_total'] + alpha * df_historico_g['num_items_subcat'])
    ) * pen_compras * pen_compras_last_year
    return df_historico_g


scaler = MinMaxScaler()
df_historico_g[['repeat_final','pop_item']] = scaler.fit_transform(df_historico_g[['repeat_final','pop_item']])
    

df_historico_g['score_final_fnn'] = (
        0.7 * df_historico_g[SCORE_COL] + 
        0.15 * df_historico_g['repeat_final'] + 
        0.15 * df_historico_g['pop_item'] #+ 
        # 0.0 * df_historico_grupby['PRIORIDAD_FINAL'].fillna(0) # Desactivado para debug
    )
    

agg_dict = {
    # Scores y features calculados (todos son idénticos en el grupo)
    SCORE_COL: 'first',
    'score_final_normalizado': 'first',
    'repeat_final': 'first',
    'pop_item': 'first',
    'score_final_fnn': 'first',

    # Features intermedios de REPEAT
    'last_buy_day': 'first',
    'count_buy': 'first',
    'count_last_day': 'first',
    'reapat_count_30': 'first',
    'reapat_count_60': 'first',
    'reapat_count_90': 'first',
    'due_B': 'first',

    # Features intermedios de POP
    'compras_item': 'first',
    'compras_subcat_total': 'first',
    'num_items_subcat': 'first',
    'compras_last_year': 'first',

    # Features de Perfil de usuario (ACTUALIZADOS)
    'recencia_hl': 'first',        # Reemplaza a recencia_hl30, hl60, score
    'freq_score': 'first',         # Reemplaza a freq_score_90d
    'sow_24m': 'first',
    'season_ratio': 'first',    
    # Promociones PACOM
    'PRIORIDAD_FINAL': 'first',
    'PRIORIDAD': 'first',
    'CODIGO_EVENTO': 'first',
    'EVENTO': 'first',
    
    # Transaccionales (agregar/promediar)
    'CANTIDAD_SUELTA': 'sum',      # Total unidades vendidas
    'PVP': 'mean',                 # Precio promedio
    'VENTA_NETA': 'mean',          # Venta promedio por transacción
    'DESCUENTO': 'mean',           # Descuento promedio
    'DIM_FACTURA': 'max',          # Última factura
    'DIM_PERIODO': 'max',          # Última fecha de compra
    'COD_LOCAL': 'first',          # Local de compra
    
    # Nombres/identificadores
    'NOMBRE_SUBCATEGORIA_x': 'first',
    'NOMBRE_SUBCATEGORIA_y': 'first'
}

# Aplicar GROUP BY (dropna=False para mantener registros sin FECHA/promoción)
df_historico_grupby_final = df_historico_grupby_final.groupby(
    ['CODIGO_FAMILIA', 'CODIGO_ITEM', 'FECHA', 'COD_SUBCATEGORIA'],
    as_index=False,
    dropna=False  # CRÍTICO: Mantener ítems sin promoción (FECHA=NaN)
).agg(agg_dict)

print(f'DataFrame después de GROUP BY: {len(df_historico_grupby_final):,} registros')
print(f'Reducción: {(1 - len(df_historico_grupby_final) / len(df_final_todas_fechas[0])) * 100:.1f}%')

# %%
# ============================================================================
# CONVERTIR COLUMNAS FLOAT A STRING PARA EVITAR NOTACIÓN CIENTÍFICA
# ============================================================================
# Crear copia para guardar
df_para_guardar = df_historico_grupby_final.copy()

# Columnas que NO deben convertirse (identificadores y enteros)
columnas_excluir = ['DIM_FACTURA', 'COD_LOCAL', 'CODIGO_FAMILIA', 'CANTIDAD_SUELTA', 
                    'CODIGO_ITEM', 'COD_SUBCATEGORIA', 'PRIORIDAD', 'CODIGO_EVENTO']

# Seleccionar columnas float EXCEPTO las excluidas
columnas_float = df_para_guardar.select_dtypes(include=['float64']).columns
columnas_a_convertir = [col for col in columnas_float if col not in columnas_excluir]

print(f'\nColumnas float encontradas: {len(columnas_float)}')
print(f'Columnas a convertir (excluyendo IDs): {len(columnas_a_convertir)}')

# Convertir solo las columnas de scores/métricas a formato fijo
for col in columnas_a_convertir:
    df_para_guardar[col] = df_para_guardar[col].apply(lambda x: f'{x:.10f}' if pd.notna(x) else x)

# CODIGO_FAMILIA: Padding de 10 dígitos con ceros a la izquierda
df_para_guardar['CODIGO_FAMILIA'] = df_para_guardar['CODIGO_FAMILIA'].str.zfill(10)

# %%
#guardar csv 
#%%
item = pd.read_excel("item.xlsx")

#%%
# ============================================================================
# CREACIÓN DE COLUMNAS TOP (3 RANKINGS)
# ============================================================================
# IMPORTANTE: Usar df_historico_grupby_final (numérico) en lugar de df_para_guardar (strings)
columnas = ['CODIGO_FAMILIA','CODIGO_ITEM', SCORE_COL,'repeat_final', 'pop_item', 'score_final_fnn','COD_SUBCATEGORIA','NOMBRE_SUBCATEGORIA_y']
df_para_guardar_cris = df_historico_grupby_final[columnas].copy()
df_para_guardar_cris.rename(columns={"CODIGO_ITEM": "COD_ITEM"}, inplace=True)
df_para_guardar_cris['COD_ITEM'] = df_para_guardar_cris['COD_ITEM'].astype('int64')

# Merge con información de división comercial
colums_unicas = ["COD_ITEM"]
item = item.drop_duplicates(subset=colums_unicas).reset_index(drop=True)
columnas_funcionales = ["COD_ITEM","NOMBRE_ITEM","COD_DIVISION_COMERCIAL","NOMBRE_DIVISION_COMERCIAL"]
item = item[columnas_funcionales].copy()
item = item[item['NOMBRE_DIVISION_COMERCIAL'].isin(['FARMA', 'CONSUMO'])]
df_para_guardar_cris = df_para_guardar_cris.merge(item, on="COD_ITEM", how="left")

# ----------------------------------------------------------------------------
# TOP 1: Ranking de SUBCATEGORÍAS por FAMILIA (ordenado por fnn_prob promedio)
# Todos los ítems de la misma subcategoría tienen el mismo TOP
# ----------------------------------------------------------------------------
# Calcular score promedio por subcategoría
df_subcategoria_score = df_para_guardar_cris.groupby(
    ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']
)[SCORE_COL].mean().reset_index()
df_subcategoria_score.rename(columns={SCORE_COL: 'score_subcat_avg'}, inplace=True)

# Ordenar subcategorías por familia y score promedio (descendente = mayor es mejor)
df_subcategoria_score = df_subcategoria_score.sort_values(
    by=['CODIGO_FAMILIA', 'score_subcat_avg'],
    ascending=[True, False]
)

# Asignar ranking a cada subcategoría
df_subcategoria_score['top_subcategoria_familia'] = df_subcategoria_score.groupby(
    'CODIGO_FAMILIA'
).cumcount() + 1

# Merge para asignar el mismo TOP a todos los ítems de la misma subcategoría
df_para_guardar_cris = df_para_guardar_cris.merge(
    df_subcategoria_score[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'top_subcategoria_familia']],
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'],
    how='left'
)

# ----------------------------------------------------------------------------
# TOP 2: Ranking GLOBAL por FAMILIA (ordenado por score_final_fnn)
# ----------------------------------------------------------------------------
df_para_guardar_cris = df_para_guardar_cris.sort_values(
    by=["CODIGO_FAMILIA", "score_final_fnn"], 
    ascending=[True, False]
)
df_para_guardar_cris['top_item_global'] = df_para_guardar_cris.groupby(
    'CODIGO_FAMILIA'
).cumcount() + 1

# ----------------------------------------------------------------------------
# TOP 3: Ranking por FAMILIA + DIVISIÓN COMERCIAL (ordenado por score_final_fnn)
# Se separa en FARMA y CONSUMO para que no se choquen los rankings
# ----------------------------------------------------------------------------
# --- FARMA ---
df_farma = df_para_guardar_cris[df_para_guardar_cris['NOMBRE_DIVISION_COMERCIAL'] == 'FARMA'].copy()
df_farma = df_farma.sort_values(by=["CODIGO_FAMILIA", "score_final_fnn"], ascending=[True, False])
df_farma['top_item_division'] = df_farma.groupby('CODIGO_FAMILIA').cumcount() + 1

# --- CONSUMO ---
df_consumo = df_para_guardar_cris[df_para_guardar_cris['NOMBRE_DIVISION_COMERCIAL'] == 'CONSUMO'].copy()
df_consumo = df_consumo.sort_values(by=["CODIGO_FAMILIA", "score_final_fnn"], ascending=[True, False])
df_consumo['top_item_division'] = df_consumo.groupby('CODIGO_FAMILIA').cumcount() + 1

# Concatenar FARMA y CONSUMO
df_concat = pd.concat([df_farma, df_consumo], ignore_index=True)


# ============================================================================
# RENOMBRAR COLUMNAS Y GUARDAR ARCHIVOS
# ============================================================================
df_final = df_concat.rename(columns={
    "CODIGO_FAMILIA": "CODIGO_FAMILIA",
    "COD_ITEM": "COD_ITEM",
    "NOMBRE_ITEM": "NOMBRE_ITEM",
    "fnn_prob": "SCORE_SUBCATEGORIA_FNN",
    "repeat_final": "RECOMPRA",
    "pop_item": "POPULARIDAD",
    "score_final_fnn": "SCORE_FINAL_FNN",
    "top_subcategoria_familia": "TOP_SUBCATEGORIA_FAMILIA",
    "top_item_global": "TOP_ITEM_GLOBAL",
    "top_item_division": "TOP_ITEM_DIVISION",
    "COD_DIVISION_COMERCIAL": "COD_DIVISION_COMERCIAL",
    "NOMBRE_DIVISION_COMERCIAL": "NOMBRE_DIVISION_COMERCIAL",
    "COD_SUBCATEGORIA": "COD_SUBCATEGORIA",
    "NOMBRE_SUBCATEGORIA_y": "NOMBRE_SUBCATEGORIA"
})

# Guardar el MISMO dataframe en Excel y CSV
df_final.to_excel("df_historico_grupby_final_general.xlsx", index=False)
df_final.to_csv('df_historico_grupby_final_general.csv', index=False)

print('\n=== ARCHIVOS GUARDADOS EXITOSAMENTE ===')
print(f'Excel: df_historico_grupby_final_general.xlsx')
print(f'CSV: df_historico_grupby_final_general.csv')
print(f'Total registros: {len(df_final):,}')
print(f'\nColumnas TOP creadas:')
print('  1. TOP_SUBCATEGORIA_FAMILIA: Ranking por familia + subcategoría (ordenado por fnn_prob)')
print('  2. TOP_ITEM_GLOBAL: Ranking por familia (ordenado por score_final_fnn)')
print('  3. TOP_ITEM_DIVISION: Ranking por familia + división comercial (ordenado por score_final_fnn)')

# %%
