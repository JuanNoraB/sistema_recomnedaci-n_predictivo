import json
import pandas as pd

#leer archivos de datos
def read_historico_test(path):
    df = pd.read_csv(path,
                     sep=';',
                     encoding='utf-8')
    df['DIM_PERIODO'] = pd.to_datetime(df['DIM_PERIODO'], format="%d-%b-%y")
    return df


def read_historico(tipo='historico', path=None):
    with open('/home/juanchx/Documents/Trabajo/SYSTEM_RECOMENDATION_FNN/config.json') as f:
        config = json.load(f)
    
    if tipo == 'historico':
        return read_historico_test(config['Path']['Historico'])
    elif tipo == 'test':
        return read_historico_test(config['Path']['Test'])
    elif tipo == 'predictions_fnn':
        return pd.read_csv(config['Path']['Result_FNN'])

    elif tipo == 'pacom':
        df_pacom = pd.read_csv(config['Path']['Pacom'], sep=',', encoding='utf-8')
        df_pacom.columns = df_pacom.columns.str.strip()
        df_pacom['FECHA'] = pd.to_datetime(df_pacom['FECHA'], format='%d-%b-%y')
        return df_pacom
    else:
        # Si se pasa un path custom
        if path is not None:
            return pd.read_csv(path)
        else:
            raise ValueError(f"Tipo '{tipo}' no reconocido. Usa 'historico', 'test', 'predictions_fnn' o 'pacom'")


