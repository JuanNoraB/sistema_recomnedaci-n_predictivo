"""
Script para calcular features hasta una fecha específica.
Se ejecuta UNA VEZ por fecha y guarda el resultado en CSV.

Uso:
    python3 compute_features.py --fecha_corte 2025-11-09
"""
import argparse
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Importar funciones de feature engineering
from feature_engineering_batch import (
    load_historical_dataset,
    compute_features_for_family
)


def main():
    parser = argparse.ArgumentParser(description='Calcular features hasta fecha específica')
    parser.add_argument('--fecha_corte', type=str, required=True,
                       help='Fecha de corte en formato YYYY-MM-DD (ej: 2025-11-09)')
    args = parser.parse_args()
    
    # Validar fecha
    try:
        fecha_corte = pd.Timestamp(args.fecha_corte)
    except:
        print(f"❌ Error: Fecha inválida '{args.fecha_corte}'. Usar formato YYYY-MM-DD")
        sys.exit(1)
    
    print("=" * 80)
    print("🔧 CÁLCULO DE FEATURES")
    print("=" * 80)
    print(f"\n📅 Fecha de corte: {fecha_corte.date()}")
    
    # Cargar histórico
    print("\n📂 Cargando histórico...")
    base_dir = Path(__file__).parent.parent
    hist_file = base_dir / "Data" / "Historico_08122025.csv"
    
    if not hist_file.exists():
        print(f"❌ Error: No se encuentra {hist_file}")
        sys.exit(1)
    
    df_hist = load_historical_dataset(hist_file)
    
    # Filtrar hasta fecha de corte
    df_hist = df_hist[df_hist['DIM_PERIODO'] <= fecha_corte].copy()
    
    print(f"   ✓ {len(df_hist)} registros hasta {fecha_corte.date()}")
    print(f"   ✓ {df_hist['CODIGO_FAMILIA'].nunique()} familias")
    print(f"   ✓ Período: {df_hist['DIM_PERIODO'].min().date()} a {df_hist['DIM_PERIODO'].max().date()}")
    
    # Calcular features por familia
    print("\n⏳ Calculando features por familia...")
    print("   (esto puede tardar varios minutos)")
    
    familias = df_hist['CODIGO_FAMILIA'].unique()
    all_features = []
    
    for i, familia_id in enumerate(familias, 1):
        if i % 100 == 0:
            print(f"   Procesando familia {i}/{len(familias)} ({i/len(familias)*100:.1f}%)")
        
        try:
            df_family = df_hist[df_hist['CODIGO_FAMILIA'] == familia_id].copy()
            
            # Calcular features para esta familia
            df_features = compute_features_for_family(
                df_family=df_family,
                familia_id=familia_id,
                fecha_corte=fecha_corte
            )
            
            if not df_features.empty:
                all_features.append(df_features)
        
        except Exception as e:
            print(f"   ⚠️  Error en familia {familia_id}: {e}")
            continue
    
    # Concatenar todas las features
    if not all_features:
        print("\n❌ Error: No se calcularon features")
        sys.exit(1)
    
    df_final = pd.concat(all_features, ignore_index=True)
    
    print(f"\n✅ Features calculadas:")
    print(f"   Total registros: {len(df_final)}")
    print(f"   Familias: {df_final['CODIGO_FAMILIA'].nunique()}")
    print(f"   Subcategorías: {df_final['COD_SUBCATEGORIA'].nunique()}")
    
    # Guardar en outputs/features/
    output_dir = base_dir / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Formato: features_YYYYMMDD.csv
    suffix = fecha_corte.strftime('%Y%m%d')
    output_file = output_dir / f"features_{suffix}.csv"
    
    df_final.to_csv(output_file, index=False)
    
    print(f"\n💾 Archivo guardado:")
    print(f"   {output_file}")
    print(f"   {len(df_final)} registros")
    print(f"   {len(df_final.columns)} columnas")
    
    # Resumen de columnas
    print(f"\n📋 Columnas incluidas:")
    feature_cols = [
        'recencia_hl', 'freq_baja', 'freq_media', 'freq_alta', 
        'cv_invertido', 'sow_24m', 'season_ratio'
    ]
    
    for col in feature_cols:
        if col in df_final.columns:
            non_null = df_final[col].notna().sum()
            print(f"   ✓ {col}: {non_null}/{len(df_final)} ({non_null/len(df_final)*100:.1f}%)")
    
    # Distribución de tipos de ciclo
    if 'Ciclos_tipo_ciclo' in df_final.columns:
        print(f"\n📊 Distribución de tipos de ciclo:")
        tipo_dist = df_final['Ciclos_tipo_ciclo'].value_counts()
        for tipo, count in tipo_dist.items():
            print(f"   {tipo:15s}: {count:5d} ({count/len(df_final)*100:4.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ FEATURES CALCULADAS Y GUARDADAS")
    print("=" * 80)
    
    print(f"\n💡 Para usar en entrenamiento:")
    print(f"   python3 src/keras/train_fnn.py --features_file {output_file.name}")
    
    print(f"\n💡 Para usar en comparación:")
    print(f"   python3 src/keras/compare_final.py --features_file {output_file.name}")


if __name__ == "__main__":
    main()
