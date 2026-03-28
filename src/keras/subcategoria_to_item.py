"""
============================================================================
SUBCATEGORÍA TO ITEM - MODELO LINEAL ITEM SCORING
============================================================================
Este script toma el score FNN por subcategoría y calcula scores a nivel item
usando características de POPULARIDAD y RECOMPRA basadas en el histórico.

INPUTS:
- predictions_fnn_final.csv: Score FNN por (FAMILIA, SUBCATEGORIA)
- Historico_08122025.csv: Ventas históricas para el calculo de POP Y RECOMPRA

OUTPUT:
- df_item_score_final.csv: 7 columnas
  * CODIGO_FAMILIA
  * COD_SUBCATEGORIA  
  * CODIGO_ITEM
  * SCORE_FNN_SUBCAT (score subcategoría del modelo FNN)
  * POPULARIDAD (score popularidad del item)
  * RECOMPRA (score recompra del item)
  * SCORE_FINAL = 0.7*score_fnn + 0.15*recompra + 0.15*popularidad
============================================================================

NUEVO SALIDA COLUMNAS:
    * CODIGO_FAMILIA                   ---> df_score_subcat CODIGO_FAMILIA
    * CODIGO_ITEM                      ---> df_historico COD_ITEM
    * SCORE_FNN_SUBCAT                 ---> df_score_subcat fnn_prob
    * RECOMPRA                         ---> NUEVA FEATURE 
    * POPULARIDAD                      ---> NUEVA FEATURE
    * SCORE_ITEM                       ---> NUEVA FEATURE     
    * COD_SUBCATEGORIA                 ---> df_score_subcat COD_SUBCATEGORIA
    * NOMBRE_SUBCATEGORIA              ---> df_historico COD_SUBCATEGORIA
    * TOP SIN NADA                      
    * NOMBRE_ITEM                      ---> df_historico NOMBRE_ITEM
    * COD_DIVISION_COMERCIAL           ---> df_historico COD_DIVISION_COMERCIAL
    * NOMBRE_DIVISION_COMERCIAL        ---> df_historico NOMBRE_DIVISION_COMERCIAL
    * TOP S NO SE QUE ES
    * TOP_SC_DC NO (SUBC/CLIENTE)      ---> NUEVA FEATEURE
    * TOP_SC (SUBC Y DIV COMERCIAL)    ---> NUEVA FEATEURE
    * VENTA                            ---> df_historico VENTA_NETA
    * UNIDADES                         ---> df_historico CANTIDAD_SUELTA
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path

# Agregar path raíz para importar file_read
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from file_read import read_historico

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Scoring de items con características PACOM')
parser.add_argument('--fecha_recomendacion', type=str, required=True,
                    help='Fecha de recomendación en formato YYYY-MM-DD')
args = parser.parse_args()

# Fecha de recomendación (para filtrar PACOM)
FECHA_RECOMENDACION = pd.Timestamp(args.fecha_recomendacion)

# Fecha de evaluación (para calcular recompra)
FECHA_CORTE = pd.Timestamp('2025-11-30')

# Fecha de referencia para validación (21 de diciembre 2025)
FECHA_VALIDACION_REF = pd.Timestamp('2025-12-21')


# Pesos del modelo lineal
PESO_SUBCAT = 0.70
PESO_RECE_CERCA = 0.10
PESO_POP = 0.10
PESO_PACOM = 0.10


# Constantes recencia cercania
M = -0.4
B = 1.6
# ============================================================================
# CARGA DE DATOS
# ============================================================================
print("\n📂 Cargando datos...")

# Scores FNN por subcategoría (desde predictions_fnn_final.csv)
df_score_subcat = read_historico(tipo='predictions_fnn')
print(f"   ✓ Predictions FNN: {len(df_score_subcat)} registros")

# Histórico de ventas
df_historico = read_historico(tipo='historico')

# Filtrar por fecha de corte
df_historico = df_historico[df_historico['DIM_PERIODO'] <= FECHA_CORTE].copy()
print(f"   ✓ Histórico: {len(df_historico)} registros")


# Filtrar por FECHA_VALIDACION_REF
# Test (disponible si se necesita)
df_test = read_historico(tipo='test')
df_test = df_test[df_test['DIM_PERIODO'] <= FECHA_VALIDACION_REF].copy()
print(f"   ✓ Test: {len(df_test)} registros")

# PACOM data
df_pacom = read_historico(tipo='pacom')
print(f"   ✓ PACOM: {len(df_pacom)} registros")

# Filtrar PACOM por fecha de recomendación
df_pacom_filtered = df_pacom[df_pacom['FECHA'] == FECHA_RECOMENDACION].copy()
print(f"   ✓ PACOM filtrado por fecha {FECHA_RECOMENDACION.date()}: {len(df_pacom_filtered)} registros")

# Seleccionar columnas necesarias y crear indicador
if len(df_pacom_filtered) > 0:
    df_pacom_filtered = df_pacom_filtered[['CODIGO_ITEM', 'COD_SUBCATEGORIA']].drop_duplicates()
    df_pacom_filtered['en_pacom'] = 1
else:
    df_pacom_filtered = pd.DataFrame(columns=['CODIGO_ITEM', 'COD_SUBCATEGORIA', 'en_pacom'])


# Seleccionar columnas necesarias del histórico
columnas_historico = [
    'CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'NOMBRE_SUBCATEGORIA', 'COD_ITEM', 'NOMBRE_ITEM',
    'DIM_PERIODO', 'DIM_FACTURA', 'VENTA_NETA', 'CANTIDAD_SUELTA', 'COD_DIVISION_COMERCIAL', 'NOMBRE_DIVISION_COMERCIAL'
]
df_historico = df_historico[columnas_historico].copy()
df_historico.rename(columns={'COD_ITEM': 'CODIGO_ITEM'}, inplace=True)


# Mantener histórico completo con transacciones para cálculos
df_historico_c = pd.merge(
    df_historico, 
    df_score_subcat, 
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'], 
    how='inner'
)
print(f"   ✓ {len(df_historico_c)} registros transaccionales después del merge")
print(f"   ✓ {df_historico_c['CODIGO_FAMILIA'].nunique()} familias")
print(f"   ✓ {df_historico_c['COD_SUBCATEGORIA'].nunique()} subcategorías")
print(f"   ✓ {df_historico_c['CODIGO_ITEM'].nunique()} items únicos")

# Filtrar histórico por FECHA_CORTE para consistencia en cálculos
print(f"   ✓ {len(df_historico_c)} registros después de filtrar por FECHA_CORTE ({FECHA_CORTE.date()})")

# Crear df a nivel único (FAMILIA, SUBCATEGORIA, ITEM) con columnas necesarias
df = df_historico_c[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM', 'NOMBRE_SUBCATEGORIA', 'NOMBRE_ITEM', 
                     'COD_DIVISION_COMERCIAL', 'NOMBRE_DIVISION_COMERCIAL', 'fnn_prob']].drop_duplicates(
    subset=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM']
).reset_index(drop=True)
print(f"   ✓ {len(df)} items únicos en df base")

# ----------------------------------------------------------------------------
# TOPS A NIVEL SUBCATEGORÍA
# ----------------------------------------------------------------------------
print("\n📊 Calculando rankings de subcategorías...")

# TOP subcategoría por FAMILIA (basado en fnn_prob)
df_subcat_tops = df[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'COD_DIVISION_COMERCIAL', 'fnn_prob']].drop_duplicates(
    subset=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']
)
df_subcat_tops['top_subcategoria'] = df_subcat_tops.groupby('CODIGO_FAMILIA')['fnn_prob'].rank(
    ascending=False, method='first'
).astype(int)

# TOP subcategoría por FAMILIA + DIVISION (basado en fnn_prob)
df_subcat_tops['top_subcategoria_division'] = df_subcat_tops.groupby(
    ['CODIGO_FAMILIA', 'COD_DIVISION_COMERCIAL']
)['fnn_prob'].rank(ascending=False, method='first').astype(int)

# Merge tops a df
df = df.merge(
    df_subcat_tops[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'top_subcategoria', 'top_subcategoria_division']],
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'],
    how='left'
)
print(f"   ✓ Rankings de subcategoría calculados")

# ============================================================================
# CÁLCULO DE CARACTERÍSTICAS POR ITEM
# ============================================================================
print("\n📊 Calculando características por item..........")

# ----------------------------------------------------------------------------
# 1. POPULARIDAD: Score de popularidad del ítem (CALCULA SOBRE TRANSACCIONES)
# ----------------------------------------------------------------------------
print("   → Calculando POPULARIDAD...........")

# Compras del ITEM específico (facturas únicas)
compras_item = df_historico_c.groupby(
    ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM']
)['DIM_FACTURA'].nunique().reset_index(name='compras_item')

# Compras TOTALES de la subcategoría (facturas únicas)
compras_subcat = df_historico_c.groupby(
    ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']
)['DIM_FACTURA'].nunique().reset_index(name='compras_subcat_total')

# Número de ítems únicos en la subcategoría
num_items_subcat = df_historico_c.groupby(
    ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']
)['CODIGO_ITEM'].nunique().reset_index(name='num_items_subcat')

# Compras del ítem en el ÚLTIMO AÑO
t_year = FECHA_CORTE - pd.Timedelta(days=365)
df_last_year = df_historico_c[df_historico_c['DIM_PERIODO'] >= t_year]
compras_last_year = df_last_year.groupby(
    ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM']
)['DIM_FACTURA'].nunique().reset_index(name='compras_last_year')

# Merge todo a df
df = df.merge(compras_item, on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM'], how='left')
df = df.merge(compras_subcat, on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'], how='left')
df = df.merge(num_items_subcat, on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'], how='left')
df = df.merge(compras_last_year, on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM'], how='left')
df['compras_last_year'] = df['compras_last_year'].fillna(0)



# Penalización 2: se ha comprado menos de 2 veces en el último año
df['pen_compras_last_year'] = np.where(
    df['compras_last_year'] < 2,
    0.5,
    1.0
)

# Pop base (frecuencia relativa en la subcategoría con Laplace smoothing)
alpha = 1
df['pop_item'] = (
    (df['compras_item'] + alpha) /
    (df['compras_subcat_total'] + alpha * df['num_items_subcat'])
) * df['pen_compras_last_year']

print(f"   ✓ Popularidad calculada para {len(df)} items únicos")

# ----------------------------------------------------------------------------
# 2. RECENCIA_CERCANIA: Calculada sobre datos transaccionales
# ----------------------------------------------------------------------------
print("\n📊 Calculando RECENCIA_CERCANIA...")

g = df_historico_c.groupby(['CODIGO_FAMILIA','COD_SUBCATEGORIA','CODIGO_ITEM'], as_index=False).agg(
    dias_ultima_compra=("DIM_PERIODO", lambda s: (FECHA_CORTE - s.max()).days),
    ciclo_dias=("ciclo_dias_mu", "min"),
    ciclo_binario=("Debug_ciclos_ciclo_binario", "min"),
    estabilidad=("cv_invertido", "min")
)

# Default (tu "else: 1")
g["recencia_cercania"] = 1.0

# Caso ciclo_binario == 0
m0 = g["ciclo_binario"].eq(0)
g.loc[m0 & g["ciclo_dias"].gt(0), "recencia_cercania"] = 0.2
g.loc[m0 & g["ciclo_dias"].eq(0), "recencia_cercania"] = 0.0

# Caso ciclo_binario == 1 y ciclo_dias > 0
m1 = g["ciclo_binario"].eq(1) & g["ciclo_dias"].gt(0)
factor = g.loc[m1, "dias_ultima_compra"] / g.loc[m1, "ciclo_dias"]

# Tramos
g.loc[m1 & (factor < 1.5), "recencia_cercania"] = 1.0
g.loc[m1 & (factor > 4.0), "recencia_cercania"] = 0.0
mid = m1 & (factor >= 1.5) & (factor <= 4.0)
g.loc[mid, "recencia_cercania"] = M * factor.loc[(factor >= 1.5) & (factor <= 4.0)] + B

# Merge recencia_cercania a df (incluir columnas de debug)
df = pd.merge(
    df,
    g[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM', 'recencia_cercania', 
       'dias_ultima_compra', 'ciclo_dias', 'ciclo_binario', 'estabilidad']],
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM'],
    how='left'
)

print(f"   ✓ Recencia cercania agregada para {len(df)} items")

# ----------------------------------------------------------------------------
# 3. PACOM: Valor adicional si se encuentra dentro de un especifico
# ----------------------------------------------------------------------------
print("\n📊 Calculando PACOM...")

# Left join para agregar indicador PACOM
df = pd.merge(
    df,
    df_pacom_filtered,
    on=['CODIGO_ITEM', 'COD_SUBCATEGORIA'],
    how='left'
)

# Llenar NaN con 0 (items que no están en PACOM)
df['en_pacom'] = df['en_pacom'].fillna(0)

print(f"   ✓ Items en PACOM para esta fecha: {df['en_pacom'].sum():.0f}")
print(f"   ✓ Total items únicos: {len(df)}")

# ----------------------------------------------------------------------------
# 4. SCORE FINAL
# ----------------------------------------------------------------------------
print("\n📊 Calculando SCORE FINAL...")

df['score_final'] = (
    PESO_SUBCAT * df['fnn_prob'] + 
    PESO_RECE_CERCA * df['recencia_cercania'] + 
    PESO_POP * df['pop_item'] +
    PESO_PACOM * df['en_pacom'] 
)

# ----------------------------------------------------------------------------
# TOP ITEM por FAMILIA
# ----------------------------------------------------------------------------
print("\n📊 Calculando ranking de items...")
df['top_item'] = df.groupby('CODIGO_FAMILIA')['score_final'].rank(
    ascending=False, method='first'
).astype(int)
print(f"   ✓ Ranking de items calculado")

# ----------------------------------------------------------------------------
# MÉTRICAS DEL PERIODO DE VALIDACIÓN (df_test)
# ----------------------------------------------------------------------------
print("\n📊 Calculando métricas de validación...")


# Preparar df_test a nivel item
df_test_agg = df_test.groupby(['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'COD_ITEM']).agg({
    'CANTIDAD_SUELTA': 'sum',
    'DIM_PERIODO': 'max'
}).reset_index()

df_test_agg.rename(columns={
    'COD_ITEM': 'CODIGO_ITEM',
    'CANTIDAD_SUELTA': 'unidades_sueltas_validacion',
    'DIM_PERIODO': 'fecha_ultima_compra_validacion'
}, inplace=True)

# Indicador de venta en validación
df_test_agg['venta_validacion'] = 1

# Calcular días desde última compra en validación
df_test_agg['dias_ultima_compra_validacion'] = (
    FECHA_VALIDACION_REF - df_test_agg['fecha_ultima_compra_validacion']
).dt.days

print(f"   ✓ {len(df_test_agg)} items únicos en periodo de validación")

# Venta a nivel SUBCATEGORÍA (FAMILIA + SUBCATEGORIA)
df_test_subcat = df_test[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']].drop_duplicates()
df_test_subcat['venta_subcategoria_validacion'] = 1
print(f"   ✓ {len(df_test_subcat)} subcategorías con venta en validación")

# Merge con df principal (LEFT JOIN)
df_final = df.merge(
    df_test_agg[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM', 
                 'venta_validacion', 'unidades_sueltas_validacion', 
                 'dias_ultima_compra_validacion', 'fecha_ultima_compra_validacion']],
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'CODIGO_ITEM'],
    how='left'
)

# Merge venta_subcategoria_validacion
df_final = df_final.merge(
    df_test_subcat,
    on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'],
    how='left'
)

# Llenar NaN con 0 para items sin venta en validación
df_final['venta_validacion'] = df_final['venta_validacion'].fillna(0).astype(int)
df_final['unidades_sueltas_validacion'] = df_final['unidades_sueltas_validacion'].fillna(0).astype(int)
df_final['dias_ultima_compra_validacion'] = df_final['dias_ultima_compra_validacion'].fillna(-1).astype(int)  # -1 indica sin compra
df_final['venta_subcategoria_validacion'] = df_final['venta_subcategoria_validacion'].fillna(0).astype(int)

print(f"   ✓ Métricas de validación agregadas")
print(f"   ✓ Items con venta en validación: {df_final['venta_validacion'].sum():.0f}")
print(f"   ✓ Subcategorías con venta en validación: {df_final['venta_subcategoria_validacion'].sum():.0f}")
print(f"   ✓ {len(df_final)} items únicos totales")

# Renombrar columnas con prefijos para debug
df_final.rename(columns={
    'fnn_prob': 'SCORE_FNN_SUBCAT',
    'pop_item': 'POPULARIDAD',
    'recencia_cercania': 'RECENCIA_CERCANIA',
    'score_final': 'SCORE_FINAL',
    # Prefijos POPULARIDAD
    'compras_item': 'pop_compras_item',
    'compras_subcat_total': 'pop_compras_subcat_total',
    'num_items_subcat': 'pop_num_items_subcat',
    'compras_last_year': 'pop_compras_last_year',
    'pen_compras': 'pop_pen_unico_item',
    'pen_compras_last_year': 'pop_pen_compras_anio',
    # Prefijos RECENCIA
    'dias_ultima_compra': 'rec_dias_ultima_compra',
    'ciclo_dias': 'rec_ciclo_dias',
    'ciclo_binario': 'rec_ciclo_binario',
    'estabilidad': 'rec_estabilidad',
    # Prefijos PACOM
    'en_pacom': 'pacom_en_lista'
}, inplace=True)

# Reordenar columnas: principales + scores + tops + división + validación + debug
columnas_principales = [
    'CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'NOMBRE_SUBCATEGORIA',
    'CODIGO_ITEM', 'NOMBRE_ITEM'
]

columnas_scores = [
    'SCORE_FNN_SUBCAT', 'RECENCIA_CERCANIA', 'POPULARIDAD', 'SCORE_FINAL'
]

columnas_recencia_debug = [
    'rec_dias_ultima_compra', 'rec_ciclo_dias', 'rec_ciclo_binario', 'rec_estabilidad'
]

columnas_popularidad_debug = [
    'pop_compras_item', 'pop_compras_subcat_total', 'pop_num_items_subcat',
    'pop_compras_last_year', 'pop_pen_compras_anio'
]

columnas_pacom = [
    'pacom_en_lista'
]

columnas_tops = [
    'top_subcategoria', 'top_subcategoria_division'
]

columnas_division = [
    'COD_DIVISION_COMERCIAL', 'NOMBRE_DIVISION_COMERCIAL'
]

columnas_top_item = [
    'top_item'
]

columnas_validacion = [
    'venta_subcategoria_validacion', 'venta_validacion', 'unidades_sueltas_validacion', 
    'dias_ultima_compra_validacion', 'fecha_ultima_compra_validacion'
]

columnas_finales = (columnas_principales + columnas_scores + 
                    columnas_recencia_debug + columnas_popularidad_debug + 
                    columnas_pacom + columnas_tops + 
                    columnas_division + columnas_top_item + columnas_validacion)

# Filtrar solo las columnas que existen en df_final
columnas_finales = [col for col in columnas_finales if col in df_final.columns]
df_final = df_final[columnas_finales]

# ============================================================================
# GUARDAR RESULTADO
# ============================================================================
print("\n💾 Guardando resultado...")

output_file = Path(__file__).parent / "df_item_score_final.csv"
df_final.to_csv(output_file, index=False)
df_final.to_excel(output_file.with_suffix(".xlsx"), index=False)

print(f"   ✓ Archivo guardado: {output_file}")
print(f"   ✓ Total registros: {len(df_final):,}")
print(f"   ✓ Columnas: {list(df_final.columns)}")

print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO")
print("=" * 80)
print(f"\nESTADÍSTICAS:")
print(f"  - Familias: {df_final['CODIGO_FAMILIA'].nunique()}")
print(f"  - Subcategorías: {df_final['COD_SUBCATEGORIA'].nunique()}")
print(f"  - Items: {len(df_final)}")
print(f"\nSCORES PROMEDIO:")
print(f"  - Score FNN Subcat: {df_final['SCORE_FNN_SUBCAT'].mean():.4f}")
print(f"  - Popularidad: {df_final['POPULARIDAD'].mean():.4f}")
print(f"  - Recencia Cercania: {df_final['RECENCIA_CERCANIA'].mean():.4f}")
print(f"  - PACOM (en_lista): {df_final['pacom_en_lista'].mean():.4f}")
print(f"  - Score Final: {df_final['SCORE_FINAL'].mean():.4f}")
print(f"\nFECHA RECOMENDACIÓN: {FECHA_RECOMENDACION.date()}")
