"""
=============================================================================
ENTRENAMIENTO FNN - SISTEMA DE RECOMENDACIÓN
=============================================================================

CONFIGURACIÓN:
- Cambia FECHA_CORTE para entrenar con diferentes períodos
- Usa with_validation=True para split 80/20 (ver overfitting)
- Usa with_validation=False para entrenar con 100% (producción)

FECHA_CORTE = '2025-11-09': Modelo conservador (menos info, mejor en dic)
FECHA_CORTE = '2025-11-30': Modelo con toda la info (más info, peor en dic)

¿Por qué Nov 9 es mejor que Nov 30?
→ Las features (recencia, freq, etc.) capturan patrones de compra
→ Si alguien compró el 30, en diciembre NO debe recomendarse (recién compró)
→ El modelo aprende: "recencia alta + frecuencia alta = NO comprar ahora"
→ Con Nov 30, el modelo ve compras muy recientes que sesgan las predicciones
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import argparse
import os
from contextlib import contextmanager
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ignorar warnings específicos para una salida más limpia
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*incompatible dtype.*')

@contextmanager
def suppress_stdout():
    """Redirige temporalmente la salida estándar a /dev/null para silenciar prints."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import matplotlib
matplotlib.use('Agg')  # No GUI
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Importar feature engineering
sys.path.append(str(Path(__file__).parent.parent))
from feature_engineering_batch import compute_features_for_family, load_historical_dataset


# =============================================================================
# CONFIGURACIÓN PRINCIPAL
# =============================================================================

FECHA_CORTE = '2025-11-09'  # '2025-11-09' o '2025-11-30'
WITH_VALIDATION = True       # True: split 80/20 | False: 100% datos


def compute_features_and_target(fecha_corte_str):
    """
    Calcula features hasta fecha_corte y target Nov 10-30
    Features: hasta fecha_corte (variable)
    """
    print("\n" + "=" * 80)
    print(f"CALCULANDO FEATURES Y TARGET")
    print("=" * 80)
    
    # 1. Cargar histórico
    hist_file = Path(__file__).parent.parent.parent / "Data" / "Historico_08122025.csv"
    print(f"\n📂 Cargando histórico: {hist_file}")
    df_historico = load_historical_dataset(hist_file)
    print(f"   ✓ {len(df_historico)} registros")
    print(f"   ✓ Fechas: {df_historico['DIM_PERIODO'].min()} a {df_historico['DIM_PERIODO'].max()}")
    
    # 2. Dividir histórico
    fecha_corte_features = pd.Timestamp(fecha_corte_str)
    fecha_inicio_target = pd.Timestamp('2025-11-10')
    fecha_fin_target = pd.Timestamp('2025-11-30')
    
    print(f"\n📊 División:")
    print(f"   Features: hasta {fecha_corte_features} ← CONFIGURABLE")
    print(f"   Target: {fecha_inicio_target} a {fecha_fin_target} (21 días compras reales)")
    
    df_features_hist = df_historico[df_historico['DIM_PERIODO'] <= fecha_corte_features].copy()
    df_target_hist = df_historico[
        (df_historico['DIM_PERIODO'] >= fecha_inicio_target) & 
        (df_historico['DIM_PERIODO'] <= fecha_fin_target)
    ].copy()
    
    print(f"   ✓ Registros para features: {len(df_features_hist)}")
    print(f"   ✓ Registros para target: {len(df_target_hist)}")
    
    # 3. Calcular features
    print(f"\n🔧 Calculando features (hasta {fecha_corte_features})...")
    familias = df_features_hist['CODIGO_FAMILIA'].unique()
    results = []
    
    for idx, familia in enumerate(familias, 1):
        if idx % 100 == 0:
            print(f"   {idx}/{len(familias)}...", end='\r')
        
        df_fam = df_features_hist[df_features_hist['CODIGO_FAMILIA'] == familia].copy()
        
        try:
            # Pasar fecha_corte explícitamente para cálculo de ciclos largos (hasta 3 años)
            features = compute_features_for_family(df_fam, familia, fecha_corte=fecha_corte_features)
            if features.empty:
                continue
            if 'nucleo' in features.columns:
                features = features.rename(columns={'nucleo': 'CODIGO_FAMILIA'})
            results.append(features)
        except Exception as e:
            # Mostrar error si es diferente al esperado
            if idx <= 3:  # Solo mostrar primeros errores para debug
                print(f"\n   ⚠️ Error en familia {familia}: {str(e)}")
            continue
    
    print()
    df_features = pd.concat(results, ignore_index=True)
    print(f"   ✓ Features: {len(df_features)} registros, {df_features['CODIGO_FAMILIA'].nunique()} familias")
    
    # 4. Calcular target
    print(f"\n🎯 Calculando target (compras {fecha_inicio_target.date()} a {fecha_fin_target.date()})...")
    compras_target = df_target_hist.groupby(['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']).size().reset_index()
    compras_target.columns = ['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'n_compras']
    compras_target['target'] = 1
    
    print(f"   ✓ Compras en target: {len(compras_target)} familia-subcategoría")
    
    # 5. Merge
    df_final = df_features.merge(
        compras_target[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA', 'target']],
        on=['CODIGO_FAMILIA', 'COD_SUBCATEGORIA'],
        how='left'
    )
    df_final['target'] = df_final['target'].fillna(0).astype(int)
    
    print(f"\n📊 Dataset completo (antes de filtrar):")
    print(f"   Total: {len(df_final)} registros")
    print(f"   Target=1: {df_final['target'].sum()} ({df_final['target'].mean()*100:.1f}%)")
    
    # Filtrar solo registros con ciclos detectados (cortos o largos)
    df_antes = len(df_final)
    target_antes = df_final['target'].sum()
    
    df_final = df_final[df_final['Ciclos_tipo_ciclo'] != 'no_ciclico'].copy()
    
    print(f"\n🎯 Dataset filtrado (solo cíclicos: cortos + largos):")
    print(f"   Total: {len(df_final)} registros (-{df_antes - len(df_final)} no_cíclicos)")
    print(f"   Target=1: {df_final['target'].sum()} ({df_final['target'].mean()*100:.1f}%)")
    print(f"   Mejora balance: {df_final['target'].mean()*100:.1f}% vs {target_antes/df_antes*100:.1f}% (+{(df_final['target'].mean() - target_antes/df_antes)*100:.1f}pp)")
    
    tipo_dist = df_final['Ciclos_tipo_ciclo'].value_counts()
    print(f"\n   Distribución:")
    for tipo in ['corto', 'largo']:
        if tipo in tipo_dist.index:
            df_tipo = df_final[df_final['Ciclos_tipo_ciclo'] == tipo]
            print(f"      {tipo:6s}: {len(df_tipo):5d} ({len(df_tipo)/len(df_final)*100:4.1f}%) - Target=1: {df_tipo['target'].sum()} ({df_tipo['target'].mean()*100:.1f}%)")
    
    return df_final


def create_model():
    """
    
    Arquitectura:
    - Input: 4 features (recencia, freq, sow, season)
    - Hidden 1: 64 neuronas + ReLU + Dropout(0.3)
    - Hidden 2: 32 neuronas + ReLU + Dropout(0.2)
    - Output: 1 neurona + Sigmoid (probabilidad 0-1)
    """
    model = models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(64, activation='relu', name='hidden1'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='hidden2'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def evaluate_top_k(model, test_df, feature_cols, scaler, k=3):
    """
    Evalúa TOP-K por familia
    
    Para cada familia:
    1. Predice probabilidad de compra para cada subcategoría
    2. Ordena por probabilidad
    3. Toma TOP-K
    4. Compara con compras reales (target=1)
    """
    print(f"\n📊 Evaluando TOP-{k} en test set...")
    
    precisions = []
    recalls = []
    hit_rates = []
    
    for familia in test_df['CODIGO_FAMILIA'].unique():
        df_fam = test_df[test_df['CODIGO_FAMILIA'] == familia].copy()
        
        #no considero las familias con compras menos a k
        if len(df_fam) < k:
            continue
        
        # Preparar features
        X = np.nan_to_num(df_fam[feature_cols].values, nan=0.0)
        X_scaled = scaler.transform(X)
        
        # Predecir
        probs = model.predict(X_scaled, verbose=0).flatten()
        df_fam['prob'] = probs
        
        # TOP-K
        top_k_items = df_fam.nlargest(k, 'prob')['COD_SUBCATEGORIA'].values
        compradas = df_fam[df_fam['target'] == 1]['COD_SUBCATEGORIA'].values
        
        if len(compradas) == 0:
            continue
        
        # Métricas
        n_correctas = len(set(top_k_items) & set(compradas))
        precision = n_correctas / k
        recall = n_correctas / len(compradas)
        hit_rate = 1.0 if n_correctas > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)
    
    results = {
        'precision@3': np.mean(precisions),
        'recall@3': np.mean(recalls),
        'hit_rate@3': np.mean(hit_rates),
        'n_families': len(precisions)
    }
    
    print(f"   Precision@{k}: {results[f'precision@{k}']:.4f} ({results[f'precision@{k}']*100:.1f}%)")
    print(f"   Recall@{k}: {results[f'recall@{k}']:.4f} ({results[f'recall@{k}']*100:.1f}%)")
    print(f"   Hit Rate@{k}: {results[f'hit_rate@{k}']:.4f} ({results[f'hit_rate@{k}']*100:.1f}%)")
    
    return results


def plot_training_history(history, suffix, with_validation=True):
    """
    Genera visualizaciones del entrenamiento
    
    1. Loss curves (train + val)
    2. Accuracy curves (si disponible)
    3. Learning rate (si cambia)
    """
    print(f"\n📊 Generando visualizaciones...")
    
    hist_dict = history.history
    epochs = range(1, len(hist_dict['loss']) + 1)
    
    # Figura con 2 subplots
    if with_validation:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]
    
    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, hist_dict['loss'], 'b-', linewidth=2, label='Train Loss')
    if with_validation and 'val_loss' in hist_dict:
        ax1.plot(epochs, hist_dict['val_loss'], 'r-', linewidth=2, label='Val Loss')
    
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
    ax1.set_title('Evolución del Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Anotaciones
    final_train = hist_dict['loss'][-1]
    ax1.annotate(f'Final: {final_train:.4f}',
                xy=(len(epochs), final_train),
                xytext=(len(epochs)*0.7, final_train*1.1),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    if with_validation and 'val_loss' in hist_dict:
        final_val = hist_dict['val_loss'][-1]
        ax1.annotate(f'Final: {final_val:.4f}',
                    xy=(len(epochs), final_val),
                    xytext=(len(epochs)*0.7, final_val*0.9),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Plot 2: Accuracy (si existe)
    if with_validation and len(axes) > 1:
        ax2 = axes[1]
        if 'accuracy' in hist_dict:
            ax2.plot(epochs, hist_dict['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
        if 'val_accuracy' in hist_dict:
            ax2.plot(epochs, hist_dict['val_accuracy'], 'r-', linewidth=2, label='Val Accuracy')
        
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Evolución del Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    filename = f'training_plot_{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ {filename}")
    
    # Plot adicional: Comparación Train vs Val Loss (más detallado)
    if with_validation and 'val_loss' in hist_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Loss
        ax.plot(epochs, hist_dict['loss'], 'b-', linewidth=2, marker='o', markersize=4, 
                label='Train Loss', alpha=0.8)
        ax.plot(epochs, hist_dict['val_loss'], 'r-', linewidth=2, marker='s', markersize=4,
                label='Val Loss', alpha=0.8)
        
        # Gap
        gap = abs(hist_dict['loss'][-1] - hist_dict['val_loss'][-1])
        ax.axhline(y=hist_dict['loss'][-1], color='b', linestyle='--', alpha=0.3)
        ax.axhline(y=hist_dict['val_loss'][-1], color='r', linestyle='--', alpha=0.3)
        
        # Anotación del gap
        mid_y = (hist_dict['loss'][-1] + hist_dict['val_loss'][-1]) / 2
        ax.annotate(f'Gap: {gap:.4f}',
                   xy=(len(epochs)*0.9, mid_y),
                   fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7),
                   ha='center')
        
        # Zona de overfitting
        if gap > 0.05:
            ax.fill_between(epochs, hist_dict['loss'], hist_dict['val_loss'],
                           where=(np.array(hist_dict['val_loss']) > np.array(hist_dict['loss'])),
                           color='red', alpha=0.1, label='Overfitting Zone')
        
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Análisis Detallado: Train vs Val Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename2 = f'training_analysis_{suffix}.png'
        plt.savefig(filename2, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ {filename2}")


def main():
    """Pipeline principal"""
    
    parser = argparse.ArgumentParser(description='Entrenar FNN')
    parser.add_argument('--fecha', type=str, default=FECHA_CORTE,
                       help='Fecha corte para features (YYYY-MM-DD)')
    parser.add_argument('--validation', action='store_true', default=WITH_VALIDATION,
                       help='Usar split 80/20 para validación')
    args = parser.parse_args()
    
    fecha_corte = args.fecha
    with_validation = args.validation
    
    print("=" * 80)
    print("🧠 ENTRENAMIENTO FNN - SISTEMA DE RECOMENDACIÓN")
    print("=" * 80)
    print(f"\n⚙️  Configuración:")
    print(f"   Fecha corte features: {fecha_corte}")
    print(f"   Target: 2025-11-10 a 2025-11-30 (siempre fijo)")
    print(f"   Modo: {'Validación (split 80/20)' if with_validation else 'Producción (100% datos)'}")
    
    # 1. Calcular features y target (silenciando la salida de prints y warnings)
    print("\n⏳  Calculando features y target (esto puede tardar)...")
    with suppress_stdout():
        df = compute_features_and_target(fecha_corte)
    print("   ✓ Dataset final generado.")
    
    # 2. Split (si es validación)
    feature_cols = ['recencia_hl', 'freq_score', 'sow_24m', 'season_ratio']
    
    X = np.nan_to_num(df[feature_cols].values, nan=0.0)
    y = df['target'].values
    
    if with_validation:
        print(f"\n🔀 Split 80/20 (validación)...")
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        train_df = df.loc[idx_train].copy()
        test_df = df.loc[idx_test].copy()
        
        print(f"   Train: {len(train_df)} ({y_train.mean()*100:.1f}% target=1)")
        print(f"   Test: {len(test_df)} ({y_test.mean()*100:.1f}% target=1)")
    else:
        print(f"\n🔀 Usando 100% datos (producción)...")
        X_train, y_train = X, y
        X_test, y_test = None, None
        test_df = None
        print(f"   Total: {len(df)} ({y_train.mean()*100:.1f}% target=1)")
    
    # 3. Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    # 4. Crear modelo
    print(f"\n🧠 Creando modelo...")
    model = create_model()
    model.summary()
    
    # 5. Callbacks
    if with_validation:
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1
        )
        callback_list = [early_stop, reduce_lr]
    else:
        early_stop = callbacks.EarlyStopping(
            monitor='loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=15,
            verbose=1
        )
        callback_list = [early_stop, reduce_lr]
    
    # 6. Entrenar
    print(f"\n🚀 Entrenando (100 épocas máx)...")
    
    if with_validation:
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=256,
            callbacks=callback_list,
            verbose=0
        )
    else:
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=256,
            callbacks=callback_list,
            verbose=1
        )
    
    # 7. Análisis
    print(f"\n📈 Análisis del Entrenamiento:")
    print(f"   Épocas: {len(history.history['loss'])}")
    print(f"   Train Loss: {history.history['loss'][0]:.4f} → {history.history['loss'][-1]:.4f}")
    
    if with_validation:
        print(f"   Val Loss: {history.history['val_loss'][0]:.4f} → {history.history['val_loss'][-1]:.4f}")
        gap = abs(history.history['loss'][-1] - history.history['val_loss'][-1])
        print(f"   Gap: {gap:.4f}", end=" ")
        if gap < 0.02:
            print("✅ Excelente (NO overfitting)")
        elif gap < 0.05:
            print("✅ Bueno")
        else:
            print("⚠️  Revisar (posible overfitting)")
    
    # 8. Evaluar TOP-K (solo si validación)
    if with_validation:
        results = evaluate_top_k(model, test_df, feature_cols, scaler, k=3)
    
    # 9. Guardar
    print(f"\n💾 Guardando modelo...")
    
    # Nombre según configuración
    suffix = fecha_corte.replace('-', '')[-4:]  # últimos 4 dígitos (MMDD)
    model_name = f'model_{suffix}.h5'
    scaler_name = f'scaler_{suffix}.pkl'
    dataset_name = f'dataset_{suffix}.csv'
    
    model.save(model_name)
    joblib.dump(scaler, scaler_name)
    df.to_csv(dataset_name, index=False)
    
    print(f"   ✓ {model_name}")
    print(f"   ✓ {scaler_name}")
    print(f"   ✓ {dataset_name}")
    
    # Historial
    hist_data = {
        'epoch': range(1, len(history.history['loss']) + 1),
        'train_loss': history.history['loss']
    }
    if with_validation:
        hist_data['val_loss'] = history.history['val_loss']
    
    hist_df = pd.DataFrame(hist_data)
    hist_df.to_csv(f'history_{suffix}.csv', index=False)
    print(f"   ✓ history_{suffix}.csv")
    
    if with_validation:
        pd.DataFrame([results]).to_csv(f'results_{suffix}.csv', index=False)
        print(f"   ✓ results_{suffix}.csv")
    
    # 10. Generar visualizaciones
    plot_training_history(history, suffix, with_validation)
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    
    print(f"\n📝 Archivos generados:")
    print(f"   - {model_name}: Modelo entrenado")
    print(f"   - {scaler_name}: Normalizador de features")
    print(f"   - {dataset_name}: Dataset usado (features + target)")
    print(f"\n💡 Uso:")
    print(f"   from tensorflow import keras")
    print(f"   model = keras.models.load_model('{model_name}')")
    print(f"   scaler = joblib.load('{scaler_name}')")


if __name__ == "__main__":
    main()
