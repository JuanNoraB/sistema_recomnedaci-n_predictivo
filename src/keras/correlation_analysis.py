"""
Análisis de correlación: fnn_prob vs 7 features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 80)
    print("ANÁLISIS DE CORRELACIÓN: FNN_PROB vs FEATURES")
    print("=" * 80)
    
    # Cargar archivo
    print("\n📂 Cargando predictions_fnn_with_target.csv...")
    df = pd.read_csv("predictions_fnn_with_target.csv")
    print(f"   ✓ {len(df)} registros")
    
    # 7 features del modelo
    features = [
        'recencia_hl',
        'freq_baja',
        'freq_media',
        'freq_alta',
        'cv_invertido',
        'sow_24m',
        'season_ratio'
    ]
    
    # Agregar fnn_prob para correlación
    cols_correlacion = features + ['fnn_prob']
    
    # Filtrar solo columnas numéricas
    df_corr = df[cols_correlacion].copy()
    
    print(f"\n📊 Calculando correlaciones...")
    print(f"   Features analizadas: {len(features)}")
    print(f"   Registros: {len(df_corr)}")
    
    # Matriz de correlación completa
    corr_matrix = df_corr.corr()
    
    # Extraer solo correlaciones con fnn_prob
    corr_with_prob = corr_matrix['fnn_prob'].drop('fnn_prob').sort_values(ascending=False)
    
    print("\n" + "=" * 80)
    print("📈 CORRELACIÓN DE CADA FEATURE CON FNN_PROB")
    print("=" * 80)
    
    print(f"\n{'Feature':<20} {'Correlación':>12} {'Barra':<30}")
    print("-" * 65)
    
    for feature, corr in corr_with_prob.items():
        # Barra visual
        bar_length = int(abs(corr) * 30)
        bar = "█" * bar_length
        sign = "+" if corr >= 0 else "-"
        
        print(f"{feature:<20} {sign}{abs(corr):>10.4f}  {bar}")
    
    # Guardar correlaciones en CSV
    corr_df = pd.DataFrame({
        'feature': corr_with_prob.index,
        'correlacion': corr_with_prob.values
    })
    corr_df.to_csv('correlaciones_fnn.csv', index=False)
    print(f"\n💾 Guardado: correlaciones_fnn.csv")
    
    # Matriz completa
    print("\n" + "=" * 80)
    print("📊 MATRIZ DE CORRELACIÓN COMPLETA")
    print("=" * 80)
    print(corr_matrix.round(4))
    
    corr_matrix.to_csv('matriz_correlacion_completa.csv')
    print(f"\n💾 Guardado: matriz_correlacion_completa.csv")
    
    # Visualización
    print("\n🎨 Generando visualización...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Heatmap de matriz completa
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=axes[0])
    axes[0].set_title('Matriz de Correlación Completa\n(Features + fnn_prob)', 
                      fontsize=14, fontweight='bold')
    
    # 2. Barras de correlación con fnn_prob
    colors = ['green' if x > 0 else 'red' for x in corr_with_prob.values]
    axes[1].barh(range(len(corr_with_prob)), corr_with_prob.values, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(corr_with_prob)))
    axes[1].set_yticklabels(corr_with_prob.index)
    axes[1].set_xlabel('Correlación', fontsize=12)
    axes[1].set_title('Correlación de cada Feature con fnn_prob', 
                      fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Agregar valores en las barras
    for i, v in enumerate(corr_with_prob.values):
        axes[1].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                    va='center', ha='left' if v > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('correlacion_features_fnn.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Guardado: correlacion_features_fnn.png")
    
    # Análisis adicional: correlación con target
    if 'target' in df.columns:
        print("\n" + "=" * 80)
        print("🎯 CORRELACIÓN CON TARGET (compras reales)")
        print("=" * 80)
        
        cols_target = features + ['fnn_prob', 'target']
        corr_target = df[cols_target].corr()['target'].drop('target').sort_values(ascending=False)
        
        print(f"\n{'Feature':<20} {'Correlación':>12} {'Barra':<30}")
        print("-" * 65)
        
        for feature, corr in corr_target.items():
            bar_length = int(abs(corr) * 30)
            bar = "█" * bar_length
            sign = "+" if corr >= 0 else "-"
            print(f"{feature:<20} {sign}{abs(corr):>10.4f}  {bar}")
        
        # Comparación lado a lado
        comparison = pd.DataFrame({
            'feature': corr_with_prob.index,
            'corr_con_fnn_prob': corr_with_prob.values,
            'corr_con_target': [corr_target[f] for f in corr_with_prob.index]
        })
        comparison.to_csv('comparacion_correlaciones.csv', index=False)
        print(f"\n💾 Guardado: comparacion_correlaciones.csv")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    
    print("\n📁 Archivos generados:")
    print("   1. correlaciones_fnn.csv - Correlación features vs fnn_prob")
    print("   2. matriz_correlacion_completa.csv - Matriz completa")
    print("   3. correlacion_features_fnn.png - Visualización")
    if 'target' in df.columns:
        print("   4. comparacion_correlaciones.csv - Comparación fnn_prob vs target")
    
    print("\n💡 Interpretación:")
    print("   - Valores cercanos a +1: correlación positiva fuerte")
    print("   - Valores cercanos a -1: correlación negativa fuerte")
    print("   - Valores cercanos a 0: sin correlación")
    print("=" * 80)


if __name__ == "__main__":
    main()
