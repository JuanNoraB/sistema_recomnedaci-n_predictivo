"""
Genera visualizaciones del entrenamiento actual
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Cargar historial
hist_df = pd.read_csv('history_validation.csv')

print("📊 Generando visualizaciones del modelo Nov 9...")

epochs = hist_df['epoch'].values
train_loss = hist_df['train_loss'].values
val_loss = hist_df['val_loss'].values

# Plot 1: Loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1 = axes[0]
ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')

ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Loss (Binary Crossentropy)', fontsize=12)
ax1.set_title('Evolución del Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Anotaciones
final_train = train_loss[-1]
final_val = val_loss[-1]

ax1.annotate(f'Final: {final_train:.4f}',
            xy=(len(epochs), final_train),
            xytext=(len(epochs)*0.7, final_train*1.1),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

ax1.annotate(f'Final: {final_val:.4f}',
            xy=(len(epochs), final_val),
            xytext=(len(epochs)*0.7, final_val*0.9),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

# Mejora
ax2 = axes[1]
mejora_train = ((train_loss[0] - train_loss) / train_loss[0] * 100)
mejora_val = ((val_loss[0] - val_loss) / val_loss[0] * 100)

ax2.plot(epochs, mejora_train, 'b-', linewidth=2, label='Train')
ax2.plot(epochs, mejora_val, 'r-', linewidth=2, label='Val')

ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Mejora (%)', fontsize=12)
ax2.set_title('Reducción del Loss (%)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Anotaciones
ax2.annotate(f'{mejora_train[-1]:.1f}%',
            xy=(len(epochs), mejora_train[-1]),
            xytext=(len(epochs)*0.7, mejora_train[-1]*1.05),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

ax2.annotate(f'{mejora_val[-1]:.1f}%',
            xy=(len(epochs), mejora_val[-1]),
            xytext=(len(epochs)*0.7, mejora_val[-1]*0.95),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("   ✓ training_plot.png")

# Plot 2: Análisis detallado
fig, ax = plt.subplots(figsize=(12, 7))

# Loss
ax.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=4, 
        label='Train Loss', alpha=0.8)
ax.plot(epochs, val_loss, 'r-', linewidth=2, marker='s', markersize=4,
        label='Val Loss', alpha=0.8)

# Gap
gap = abs(train_loss[-1] - val_loss[-1])
ax.axhline(y=train_loss[-1], color='b', linestyle='--', alpha=0.3)
ax.axhline(y=val_loss[-1], color='r', linestyle='--', alpha=0.3)

# Anotación del gap
mid_y = (train_loss[-1] + val_loss[-1]) / 2
gap_text = f'Gap Final: {gap:.4f}\n'
if gap < 0.02:
    gap_text += '✅ EXCELENTE'
    color = 'lightgreen'
elif gap < 0.05:
    gap_text += '✅ BUENO'
    color = 'lightyellow'
else:
    gap_text += '⚠️ REVISAR'
    color = 'lightcoral'

ax.annotate(gap_text,
           xy=(len(epochs)*0.85, mid_y),
           fontsize=12,
           bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.8),
           ha='center',
           fontweight='bold')

# Estadísticas
stats_text = f'Inicio:\nTrain: {train_loss[0]:.4f}\nVal: {val_loss[0]:.4f}\n\n'
stats_text += f'Final:\nTrain: {train_loss[-1]:.4f}\nVal: {val_loss[-1]:.4f}\n\n'
stats_text += f'Mejora:\nTrain: {mejora_train[-1]:.1f}%\nVal: {mejora_val[-1]:.1f}%'

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Análisis Detallado: Train vs Val Loss (Modelo Nov 9)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("   ✓ training_analysis.png")

# Plot 3: Comparación por fase
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fase 1: Primeras 10 épocas
ax1 = axes[0, 0]
mask = epochs <= 10
ax1.plot(epochs[mask], train_loss[mask], 'b-', linewidth=2, marker='o', label='Train')
ax1.plot(epochs[mask], val_loss[mask], 'r-', linewidth=2, marker='s', label='Val')
ax1.set_title('Fase Inicial (épocas 1-10)', fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Fase 2: Épocas 10-30
ax2 = axes[0, 1]
mask = (epochs >= 10) & (epochs <= 30)
ax2.plot(epochs[mask], train_loss[mask], 'b-', linewidth=2, marker='o', label='Train')
ax2.plot(epochs[mask], val_loss[mask], 'r-', linewidth=2, marker='s', label='Val')
ax2.set_title('Fase Media (épocas 10-30)', fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Fase 3: Últimas épocas
ax3 = axes[1, 0]
mask = epochs >= (len(epochs) - 15)
ax3.plot(epochs[mask], train_loss[mask], 'b-', linewidth=2, marker='o', label='Train')
ax3.plot(epochs[mask], val_loss[mask], 'r-', linewidth=2, marker='s', label='Val')
ax3.set_title(f'Fase Final (últimas 15 épocas)', fontweight='bold')
ax3.set_xlabel('Época')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Gap evolution
ax4 = axes[1, 1]
gaps = np.abs(train_loss - val_loss)
ax4.plot(epochs, gaps, 'g-', linewidth=2, marker='d', markersize=4)
ax4.fill_between(epochs, 0, gaps, alpha=0.3, color='green')
ax4.axhline(y=0.02, color='orange', linestyle='--', label='Límite Excelente (0.02)')
ax4.axhline(y=0.05, color='red', linestyle='--', label='Límite Bueno (0.05)')
ax4.set_title('Evolución del Gap (|Train - Val|)', fontweight='bold')
ax4.set_xlabel('Época')
ax4.set_ylabel('Gap')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_phases.png', dpi=150, bbox_inches='tight')
plt.close()

print("   ✓ training_phases.png")

print("\n✅ Visualizaciones generadas:")
print("   1. training_plot.png: Loss + Mejora %")
print("   2. training_analysis.png: Análisis detallado con Gap")
print("   3. training_phases.png: Evolución por fases")
