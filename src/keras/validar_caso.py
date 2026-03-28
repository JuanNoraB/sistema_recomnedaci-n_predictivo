#!/usr/bin/env python3
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import file_read

# ============================================================================
# VALORES DE DEBUG (TU INPUT)
# ============================================================================
# CASO
CODIGO_FAMILIA = 1719363028
COD_SUBCATEGORIA = 9278
CODIGO_ITEM = 100160139

# RECENCIA
rec_dias_ultima_compra = 4
rec_ciclo_dias = 20
rec_ciclo_binario = 1
rec_estabilidad = 0.398356161303072

# POPULARIDAD
pop_compras_item = 20
pop_compras_subcat_total = 26
pop_num_items_subcat = 7
pop_compras_last_year = 18
pop_pen_compras_anio = 1

# SCORE FNN
score_fnn_subcat = 0.50965786

# PACOM
pacom_en_lista = 0

# VALORES ESPERADOS
recencia_esperada = 1.0
popularidad_esperada = 0.636363636363636
score_final_esperado = 0.520396865636364

# ============================================================================
# CONSTANTES
# ============================================================================
PESO_SUBCAT = 0.70
PESO_RECE_CERCA = 0.10
PESO_POP = 0.10
PESO_PACOM = 0.10
M = -0.4
B = 1.6
FECHA_VALIDACION_REF = pd.Timestamp('2025-12-21')

# ============================================================================
# CALCULAR RECENCIA_CERCANIA
# ============================================================================
recencia_calculada = 1.0  # Default

if rec_ciclo_binario == 0:
    if rec_ciclo_dias > 0:
        recencia_calculada = 0.2
    else:
        recencia_calculada = 0.0
elif rec_ciclo_binario == 1 and rec_ciclo_dias > 0:
    factor = rec_dias_ultima_compra / rec_ciclo_dias
    if factor < 1.5:
        recencia_calculada = 1.0
    elif factor > 4.0:
        recencia_calculada = 0.0
    else:
        recencia_calculada = M * factor + B

# ============================================================================
# CALCULAR POPULARIDAD
# ============================================================================
alpha = 1
pop_pen = 0.5 if pop_compras_last_year < 2 else 1.0

popularidad_calculada = ((pop_compras_item + alpha) / 
                         (pop_compras_subcat_total + alpha * pop_num_items_subcat)) * pop_pen

# ============================================================================
# CALCULAR SCORE FINAL
# ============================================================================
score_final_calculado = (
    PESO_SUBCAT * score_fnn_subcat +
    PESO_RECE_CERCA * recencia_calculada +
    PESO_POP * popularidad_calculada +
    PESO_PACOM * pacom_en_lista
)

# ============================================================================
# VALIDAR CON DF_TEST
# ============================================================================
df_test = file_read.read_historico(tipo='test')
df_test = df_test[df_test['DIM_PERIODO'] <= FECHA_VALIDACION_REF].copy()

# Subcategoría
filtro_subcat = (df_test['CODIGO_FAMILIA'] == CODIGO_FAMILIA) & \
                (df_test['COD_SUBCATEGORIA'] == COD_SUBCATEGORIA)
df_test_subcat = df_test[filtro_subcat]
venta_subcategoria_validacion = 1 if len(df_test_subcat) > 0 else 0

# Item
filtro_item = filtro_subcat & (df_test['COD_ITEM'] == CODIGO_ITEM)
df_test_item = df_test[filtro_item]
venta_validacion = 1 if len(df_test_item) > 0 else 0

if venta_validacion == 1:
    unidades_sueltas_validacion = int(df_test_item['CANTIDAD_SUELTA'].sum())
    fecha_ultima_compra_validacion = df_test_item['DIM_PERIODO'].max()
    dias_ultima_compra_validacion = (FECHA_VALIDACION_REF - fecha_ultima_compra_validacion).days
else:
    unidades_sueltas_validacion = 0
    fecha_ultima_compra_validacion = None
    dias_ultima_compra_validacion = -1

# ============================================================================
# OUTPUT
# ============================================================================
print(f"\nCASO: {CODIGO_FAMILIA} | {COD_SUBCATEGORIA} | {CODIGO_ITEM}")
print("=" * 80)

print(f"\n📥 VALORES DE ENTRADA:")
print(f"SCORE_FNN_SUBCAT: {score_fnn_subcat}")
print(f"pacom_en_lista: {pacom_en_lista}")

print(f"\nRECENCIA_CERCANIA")
print(f"Calculada: {recencia_calculada}")
print(f"Esperada:  {recencia_esperada}")
print(f" OK" if abs(recencia_calculada - recencia_esperada) < 0.0001 else "❌ ERROR")
print("-" * 80)
print(f"rec_dias_ultima_compra\trec_ciclo_dias\trec_ciclo_binario\trec_estabilidad")
print(f"{rec_dias_ultima_compra}\t\t\t{rec_ciclo_dias}\t\t{rec_ciclo_binario}\t\t{rec_estabilidad}")
print("-" * 80)

print(f"\nPOPULARIDAD")
print(f"Calculada: {popularidad_calculada}")
print(f"Esperada:  {popularidad_esperada}")
print(f" OK" if abs(popularidad_calculada - popularidad_esperada) < 0.0001 else "❌ ERROR")
print("-" * 80)
print(f"pop_compras_item\tpop_compras_subcat_total\tpop_num_items_subcat\tpop_compras_last_year\tpop_pen_compras_anio")
print(f"{pop_compras_item}\t\t\t{pop_compras_subcat_total}\t\t\t\t{pop_num_items_subcat}\t\t\t{pop_compras_last_year}\t\t\t{pop_pen_compras_anio}")
print("-" * 80)

print(f"\nSCORE_FINAL")
print(f"Calculado: {score_final_calculado}")
print(f"Esperado:  {score_final_esperado}")
print(f" OK" if abs(score_final_calculado - score_final_esperado) < 0.0001 else "❌ ERROR")

print(f"\nVALIDACIÓN")
print(f"venta_subcategoria_validacion: {venta_subcategoria_validacion}")
print(f"venta_validacion: {venta_validacion}")
print(f"unidades_sueltas_validacion: {unidades_sueltas_validacion}")
print(f"dias_ultima_compra_validacion: {dias_ultima_compra_validacion}")
if fecha_ultima_compra_validacion:
    print(f"fecha_ultima_compra_validacion: {fecha_ultima_compra_validacion.date()}")
