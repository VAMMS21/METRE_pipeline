import pandas as pd
import numpy as np
import os

OUTPUT_DIR = './output'
MIMIC_HDF5 = os.path.join(OUTPUT_DIR, 'MIMIC_split.hdf5')
EICU_HDF5  = os.path.join(OUTPUT_DIR, 'eICU_split.hdf5')

def hdf5_to_npy(hdf5_path, out_path):
    print(f'A converter {hdf5_path}...')
    
    vital_train = pd.read_hdf(hdf5_path, key='vital_train')
    vital_dev   = pd.read_hdf(hdf5_path, key='vital_dev')
    vital_test  = pd.read_hdf(hdf5_path, key='vital_test')
    inv_train   = pd.read_hdf(hdf5_path, key='inv_train')
    inv_dev     = pd.read_hdf(hdf5_path, key='inv_dev')
    inv_test    = pd.read_hdf(hdf5_path, key='inv_test')
    static_train = pd.read_hdf(hdf5_path, key='static_train')
    static_dev   = pd.read_hdf(hdf5_path, key='static_dev')
    static_test  = pd.read_hdf(hdf5_path, key='static_test')

    # ver colunas disponíveis
    print('Colunas static_train:', static_train.columns.tolist())
    print('Shape vital_train:', vital_train.shape)
    print('Shape static_train:', static_train.shape)

    # identificar coluna de mortalidade
    mort_col = 'hosp_mort' if 'hosp_mort' in static_train.columns else static_train.columns[0]
    print(f'A usar coluna de mortalidade: {mort_col}')

    # agrupar vital por paciente (stay_id) -> lista de arrays (n_features, n_horas)
    def build_head(vital_df):
        head = []
        for stay_id, group in vital_df.groupby(level='stay_id'):
            # pivot: linhas=horas, colunas=features -> transpor para (features, horas)
            arr = group.values.T  # shape (n_features, n_horas)
            head.append(arr.astype(np.float32))
        return head

    print('A construir train_head...')
    train_head = build_head(vital_train)
    print('A construir dev_head...')
    dev_head   = build_head(vital_dev)
    print('A construir test_head...')
    test_head  = build_head(vital_test)

    # static filter: mortalidade + outras features estáticas numéricas
    def build_static(static_df, mort_col):
        result = []
        for stay_id, row in static_df.iterrows():
            arr = row.values.astype(np.float32)
            result.append(arr)
        return result

    static_train_filter = build_static(static_train, mort_col)
    static_dev_filter   = build_static(static_dev, mort_col)
    static_test_filter  = build_static(static_test, mort_col)

    data_label = {
        'train_head': train_head,
        'static_train_filter': static_train_filter,
        'dev_head': dev_head,
        'static_dev_filter': static_dev_filter,
        'test_head': test_head,
        'static_test_filter': static_test_filter,
    }

    np.save(out_path, data_label)
    print(f'Guardado em {out_path}')
    print(f'  train: {len(train_head)} pacientes')
    print(f'  dev:   {len(dev_head)} pacientes')
    print(f'  test:  {len(test_head)} pacientes')

hdf5_to_npy(MIMIC_HDF5, os.path.join(OUTPUT_DIR, 'MIMIC_compile.npy'))
hdf5_to_npy(EICU_HDF5,  os.path.join(OUTPUT_DIR, 'eICU_compile.npy'))

print('\nConversão completa.')