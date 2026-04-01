import pandas as pd
import numpy as np
import os

print("A iniciar conversão...")

OUTPUT_DIR = './output'
MIMIC_HDF5 = os.path.join(OUTPUT_DIR, 'MIMIC_split.hdf5')
EICU_HDF5  = os.path.join(OUTPUT_DIR, 'eICU_split.hdf5')

def hdf5_to_npy(hdf5_path, out_path):
    print(f'A ler {hdf5_path}...')

    vital_train  = pd.read_hdf(hdf5_path, key='vital_train')
    vital_dev    = pd.read_hdf(hdf5_path, key='vital_dev')
    vital_test   = pd.read_hdf(hdf5_path, key='vital_test')
    static_train = pd.read_hdf(hdf5_path, key='static_train')
    static_dev   = pd.read_hdf(hdf5_path, key='static_dev')
    static_test  = pd.read_hdf(hdf5_path, key='static_test')

    # detectar nome do index de paciente automaticamente
    id_col = vital_train.index.names[0]
    print(f'Index de paciente: {id_col}')

    def build_head(vital_df):
        head = []
        for _, group in vital_df.groupby(level=id_col):
            arr = group.values.T.astype(np.float32)
            head.append(arr)
        return head

    def build_static(static_df):
        df = static_df.copy()
        dt_cols = df.select_dtypes(include=['datetime64']).columns
        df = df.drop(columns=dt_cols)
        str_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in str_cols:
            df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
        df = df.astype(np.float32)
        return [row.values for _, row in df.iterrows()]

    print('A construir arrays...')
    data_label = {
        'train_head': build_head(vital_train),
        'static_train_filter': build_static(static_train),
        'dev_head': build_head(vital_dev),
        'static_dev_filter': build_static(static_dev),
        'test_head': build_head(vital_test),
        'static_test_filter': build_static(static_test),
    }

    np.save(out_path, data_label)
    print(f'Guardado: {out_path} ({len(data_label["train_head"])} pacientes treino)')

hdf5_to_npy(MIMIC_HDF5, os.path.join(OUTPUT_DIR, 'MIMIC_compile.npy'))
hdf5_to_npy(EICU_HDF5,  os.path.join(OUTPUT_DIR, 'eICU_compile.npy'))
print('Conversão completa.')
