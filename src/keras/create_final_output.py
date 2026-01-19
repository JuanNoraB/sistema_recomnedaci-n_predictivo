"""
Script simple para generar archivo final con features FNN + target
"""
import pandas as pd
from pathlib import Path

def main():
    print("=" * 80)
    print("GENERANDO ARCHIVO FINAL")
    print("=" * 80)
    
    # Rutas
    base_dir = Path(__file__).parent.parent.parent
    historico_file = base_dir / "Data" / "Historico_08122025.csv"
    test_file = base_dir / "Data" / "data_test.csv"
    predictions_file = Path(__file__).parent / "predictions_fnn_final.csv"
    
    # =========================================================================
    # 1. NOMBRES DE SUBCATEGORÍAS (desde histórico)
    # =========================================================================
    print("\n📂 1. Cargando nombres de subcategorías...")
    df_hist = pd.read_csv(historico_file, sep=';', encoding='utf-8')
    
    # Extraer solo COD_SUBCATEGORIA y NOMBRE_SUBCATEGORIA
    nombres_subcat = df_hist[['COD_SUBCATEGORIA', 'NOMBRE_SUBCATEGORIA']].drop_duplicates()
    print(f"   ✓ {len(nombres_subcat)} subcategorías únicas")
    
    # =========================================================================
    # 2. TARGET (compras en test - primeros 21 días dic)
    # =========================================================================
    print("\n📂 2. Cargando compras reales (test - primeros 21 días)...")
    df_test = pd.read_csv(test_file, sep=';', encoding='utf-8')
    df_test['DIM_PERIODO'] = pd.to_datetime(df_test['DIM_PERIODO'])
    
    # Filtrar primeros 21 días de diciembre
    fecha_limite = pd.Timestamp('2025-12-21')
    df_test = df_test[df_test['DIM_PERIODO'] <= fecha_limite].copy()
    
    print(f"   ✓ {len(df_test)} compras en primeros 21 días")
    print(f"   ✓ Período: {df_test['DIM_PERIODO'].min()} a {df_test['DIM_PERIODO'].max()}")
    
    # GroupBy por familia y subcategoría (si compró = 1)
    compras_test = df_test.groupby(['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']).size().reset_index()
    compras_test.columns = ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'n_compras']
    compras_test['target'] = 1
    
    print(f"   ✓ {len(compras_test)} familia-subcategoría compraron")
    
    # =========================================================================
    # 3. PREDICCIONES FNN
    # =========================================================================
    print("\n📂 3. Cargando predicciones FNN...")
    df_pred = pd.read_csv(predictions_file)
    
    print(f"   ✓ {len(df_pred)} predicciones")
    print(f"   ✓ {df_pred['CODIGO_FAMILIA'].nunique()} familias")
    
    # =========================================================================
    # 4. MERGE 1: Agregar nombres de subcategorías
    # =========================================================================
    print("\n🔗 4. Agregando nombres de subcategorías (left join)...")
    df_final = df_pred.merge(nombres_subcat, on='COD_SUBCATEGORIA', how='left')
    print(f"   ✓ {len(df_final)} registros")
    
    # =========================================================================
    # 5. MERGE 2: Agregar target (compras en test)
    # =========================================================================
    print("\n🔗 5. Agregando target (compras en test - left join)...")
    df_final = df_final.merge(
        compras_test[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'target']], 
        on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'], 
        how='left'
    )
    df_final['target'] = df_final['target'].fillna(0).astype(int)
    
    print(f"   ✓ {len(df_final)} registros")
    print(f"   ✓ Target=1: {df_final['target'].sum()} ({df_final['target'].mean()*100:.1f}%)")
    
    # =========================================================================
    # 6. DESCOMPONER CICLOS EN COLUMNAS SEPARADAS
    # =========================================================================
    print("\n📋 6. Descomponiendo ciclos en columnas numéricas...")
    
    # Convertir Ciclos_ciclo_dias de string a lista y extraer valores
    import ast
    
    def parse_ciclo_dias(ciclo_str):
        """Convierte string '[min, prom, max]' a 3 valores numéricos"""
        try:
            if pd.isna(ciclo_str):
                return 0, 0, 0
            if isinstance(ciclo_str, str):
                ciclo_list = ast.literal_eval(ciclo_str)
            else:
                ciclo_list = ciclo_str
            
            if isinstance(ciclo_list, list) and len(ciclo_list) == 3:
                return ciclo_list[0], ciclo_list[1], ciclo_list[2]
            return 0, 0, 0
        except:
            return 0, 0, 0
    
    # Descomponer en 3 columnas
    df_final[['ciclo_min', 'ciclo_promedio', 'ciclo_max']] = df_final['Ciclos_ciclo_dias'].apply(
        lambda x: pd.Series(parse_ciclo_dias(x))
    )
    
    print(f"   ✓ Ciclos descompuestos en 3 columnas numéricas")
    
    # =========================================================================
    # 7. SELECCIONAR COLUMNAS FINALES
    # =========================================================================
    print("\n📋 7. Seleccionando columnas finales...")
    
    columnas_finales = [
        # Identificadores
        'CODIGO_FAMILIA',
        'COD_SUBCATEGORIA',
        'NOMBRE_SUBCATEGORIA',
        
        # 7 Features del modelo
        'recencia_hl',
        'freq_baja',
        'freq_media',
        'freq_alta',
        'cv_invertido',
        'Ciclos_cv',
        'sow_24m',
        'season_ratio',
        
        # Ciclos (descompuestos)
        'ciclo_min',                      # Ciclo inferior
        'ciclo_promedio',                 # Ciclo promedio
        'ciclo_max',                      # Ciclo superior
        'Ciclos_ciclo_dias',              # Ciclo días
        'Ciclos_gaps_normalizados',       # Gaps normalizados
        'Ciclos_tipo_ciclo',              # Tipo de ciclo
        
        # Score del modelo
        'fnn_prob',
        
        # Target
        'target'
    ]
    
    # Verificar qué columnas existen
    columnas_existentes = [c for c in columnas_finales if c in df_final.columns]
    columnas_faltantes = [c for c in columnas_finales if c not in df_final.columns]
    
    print(f"   ✅ Existen: {len(columnas_existentes)}")
    if columnas_faltantes:
        print(f"   ⚠️  Faltan: {columnas_faltantes}")
        print(f"      Usando solo las existentes...")
    
    df_output = df_final[columnas_existentes].copy()
    
    # =========================================================================
    # 8. GUARDAR
    # =========================================================================
    print("\n💾 8. Guardando archivo final...")
    output_file = "predictions_fnn_with_target.csv"
    df_output.to_csv(output_file, index=False)
    
    print(f"   ✓ Guardado: {output_file}")
    print(f"   ✓ {len(df_output)} registros")
    print(f"   ✓ {len(df_output.columns)} columnas")
    
    # =========================================================================
    # 9. RESUMEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ ARCHIVO FINAL GENERADO")
    print("=" * 80)
    
    print(f"\n📊 Resumen:")
    print(f"   Total registros: {len(df_output)}")
    print(f"   Familias: {df_output['CODIGO_FAMILIA'].nunique()}")
    print(f"   Subcategorías: {df_output['COD_SUBCATEGORIA'].nunique()}")
    print(f"   Target=1 (comprados): {df_output['target'].sum()} ({df_output['target'].mean()*100:.1f}%)")
    print(f"   Target=0 (no comprados): {(df_output['target']==0).sum()} ({(df_output['target']==0).mean()*100:.1f}%)")
    
    print(f"\n📋 Columnas incluidas:")
    for i, col in enumerate(df_output.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print(f"\n💡 Archivo guardado en: {Path(output_file).absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
