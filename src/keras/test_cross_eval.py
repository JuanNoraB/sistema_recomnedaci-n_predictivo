"""
Prueba cruzada: ¿Qué afecta el performance?

Test 1: Modelo Nov 9 + Eval Nov 9 = ?
Test 2: Modelo Nov 9 + Eval Nov 30 = ?
Test 3: Modelo Nov 30 + Eval Nov 30 = ?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

from tensorflow import keras

sys.path.append(str(Path(__file__).parent.parent))
from feature_engineering_batch import compute_features_for_family, load_historical_dataset


def load_test_data():
    """Carga compras reales Dic 1-9"""
    df = pd.read_excel(Path(__file__).parent.parent.parent / "Data" / "data_test.xlsx")
    df['DIM_PERIODO'] = pd.to_datetime(df['DIM_PERIODO'])
    return df


def compute_features(fecha_corte_str):
    """Calcula features hasta fecha"""
    hist_file = Path(__file__).parent.parent.parent / "Data" / "Historico_08122025.csv"
    df_historico = load_historical_dataset(hist_file)
    
    fecha_corte = pd.Timestamp(fecha_corte_str)
    df_hist_filtered = df_historico[df_historico['DIM_PERIODO'] <= fecha_corte].copy()
    
    familias = df_hist_filtered['CODIGO_FAMILIA'].unique()
    results = []
    
    for idx, familia in enumerate(familias, 1):
        if idx % 100 == 0:
            print(f"   {idx}/{len(familias)}...", end='\r')
        
        df_fam = df_hist_filtered[df_hist_filtered['CODIGO_FAMILIA'] == familia].copy()
        
        try:
            features = compute_features_for_family(df_fam, familia)
            if features.empty:
                continue
            if 'nucleo' in features.columns:
                features = features.rename(columns={'nucleo': 'CODIGO_FAMILIA'})
            results.append(features)
        except:
            continue
    
    print()
    return pd.concat(results, ignore_index=True)


def evaluate(fnn_df, test_df, top_k=3):
    """Evalúa TOP-K"""
    precisions = []
    
    for familia in test_df['CODIGO_FAMILIA'].unique():
        df_pred = fnn_df[fnn_df['CODIGO_FAMILIA'] == familia].copy()
        
        if len(df_pred) == 0:
            continue
        
        top_k_items = df_pred.nlargest(top_k, 'fnn_prob')['COD_SUBCATEGORIA'].values
        compradas = test_df[test_df['CODIGO_FAMILIA'] == familia]['COD_SUBCATEGORIA'].unique()
        
        if len(compradas) == 0:
            continue
        
        n_correctas = len(set(top_k_items) & set(compradas))
        precisions.append(n_correctas / top_k)
    
    return np.mean(precisions)


print("=" * 80)
print("TEST CRUZADO: ¿Modelo o Evaluación?")
print("=" * 80)

test_df = load_test_data()
print(f"\n✓ Test data: {len(test_df)} compras, {test_df['CODIGO_FAMILIA'].nunique()} familias")

# Test 1: Modelo Nov 9 + Eval Nov 9
print("\n" + "=" * 80)
print("TEST 1: Modelo Nov 9 + Eval Nov 9")
print("=" * 80)

print("Calculando features Nov 9...")
features_nov9 = compute_features('2025-11-09')
print(f"✓ {len(features_nov9)} registros")

print("Cargando modelo Nov 9...")
model_nov9 = keras.models.load_model('model_1109.h5')
scaler_nov9 = joblib.load('scaler_1109.pkl')

feature_cols = ['recencia_hl', 'freq_score', 'sow_24m', 'season_ratio']
X = np.nan_to_num(features_nov9[feature_cols].values, nan=0.0)
X_scaled = scaler_nov9.transform(X)

probs = model_nov9.predict(X_scaled, verbose=0).flatten()
features_nov9['fnn_prob'] = probs

precision1 = evaluate(features_nov9, test_df)
print(f"\n✅ Precision@3: {precision1:.4f} ({precision1*100:.1f}%)")


# Test 2: Modelo Nov 9 + Eval Nov 30
print("\n" + "=" * 80)
print("TEST 2: Modelo Nov 9 + Eval Nov 30 (features diferentes)")
print("=" * 80)

print("Calculando features Nov 30...")
features_nov30 = compute_features('2025-11-30')
print(f"✓ {len(features_nov30)} registros")

print("Usando MISMO modelo Nov 9...")
X = np.nan_to_num(features_nov30[feature_cols].values, nan=0.0)
X_scaled = scaler_nov9.transform(X)  # MISMO scaler

probs = model_nov9.predict(X_scaled, verbose=0).flatten()
features_nov30['fnn_prob'] = probs

precision2 = evaluate(features_nov30, test_df)
print(f"\n✅ Precision@3: {precision2:.4f} ({precision2*100:.1f}%)")


# Test 3: Modelo Nov 30 + Eval Nov 30
print("\n" + "=" * 80)
print("TEST 3: Modelo Nov 30 + Eval Nov 30")
print("=" * 80)

print("Cargando modelo Nov 30...")
model_nov30 = keras.models.load_model('model_1130.h5')
scaler_nov30 = joblib.load('scaler_1130.pkl')

print("Usando features Nov 30 (ya calculadas)...")
X = np.nan_to_num(features_nov30[feature_cols].values, nan=0.0)
X_scaled = scaler_nov30.transform(X)  # Scaler Nov 30

probs = model_nov30.predict(X_scaled, verbose=0).flatten()
features_nov30['fnn_prob'] = probs

precision3 = evaluate(features_nov30, test_df)
print(f"\n✅ Precision@3: {precision3:.4f} ({precision3*100:.1f}%)")


# Comparación
print("\n" + "=" * 80)
print("📊 COMPARACIÓN FINAL")
print("=" * 80)

results = pd.DataFrame([
    {'Test': 'Modelo Nov 9 + Eval Nov 9', 'Precision@3': f'{precision1*100:.1f}%'},
    {'Test': 'Modelo Nov 9 + Eval Nov 30', 'Precision@3': f'{precision2*100:.1f}%'},
    {'Test': 'Modelo Nov 30 + Eval Nov 30', 'Precision@3': f'{precision3*100:.1f}%'}
])

print(f"\n{results.to_string(index=False)}")

print("\n🔍 Análisis:")
if abs(precision1 - precision2) < 0.02:
    print("   Las features (Nov 9 vs Nov 30) NO afectan mucho ✅")
else:
    print(f"   Las features SÍ afectan: {abs(precision1-precision2)*100:.1f}% diferencia ⚠️")

if abs(precision2 - precision3) > 0.05:
    print(f"   El MODELO entrenado con Nov 30 es PEOR: {abs(precision2-precision3)*100:.1f}% diferencia ❌")
else:
    print("   El modelo Nov 30 no es tan malo ✅")

print("\n" + "=" * 80)
