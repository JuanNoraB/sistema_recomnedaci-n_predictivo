# %%
##generar dataset para fnn
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("/home/juanchx/Documents/Trabajo/SYSTEM_RECOMENDATION_FNN/Data/Historico_08122025.csv")
# %%
