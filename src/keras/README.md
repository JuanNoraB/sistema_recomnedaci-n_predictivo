# 🧠 Sistema de Recomendación FNN - Keras

## 📁 Archivos

### Scripts Principales:

- **`train_fnn.py`**: Entrena modelo FNN (configurable)
- **`compare_final.py`**: Compara Linear vs FNN

### Modelos Generados:

- **`model_1109.h5`**: Modelo Nov 9 (recomendado ✅)
- **`scaler_1109.pkl`**: Normalizador Nov 9
- **`dataset_1109.csv`**: Dataset usado Nov 9

- **`model_1130.h5`**: Modelo Nov 30
- **`scaler_1130.pkl`**: Normalizador Nov 30
- **`dataset_1130.csv`**: Dataset usado Nov 30

---

## 🚀 Uso Rápido

### 1. Entrenar Modelo

```bash
# Modelo Nov 9 (recomendado)
python train_fnn.py --fecha 2025-11-09 --validation

# Modelo Nov 30
python train_fnn.py --fecha 2025-11-30 --validation
```

### 2. Comparar con Linear

```bash
# Comparar Nov 9
python compare_final.py --fecha 2025-11-09

# Comparar Nov 30
python compare_final.py --fecha 2025-11-30
```

---

## 📊 Resultados

| Modelo | Precision@3 | Mejora vs Linear |
|--------|-------------|------------------|
| Linear | 18.8% | - |
| FNN (Nov 9) | **33.8%** | **+79.7%** ✅ |
| FNN (Nov 30) | 19.7% | +7.5% ⚠️ |

**Recomendación:** Usar modelo Nov 9 para producción.

---

## 📖 Documentación Completa

Lee: `/EXPLICACION_COMPLETA.md` para entender TODO el sistema a detalle.
