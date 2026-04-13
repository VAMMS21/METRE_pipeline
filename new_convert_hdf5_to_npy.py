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

    print('Colunas vital_train:', vital_train.shape[1])
    print('Colunas inv_train:', inv_train.columns.tolist())
    print('Colunas static_train:', static_train.columns.tolist())

    # identificar coluna de mortalidade e id
    mort_col = 'hosp_mort' if 'hosp_mort' in static_train.columns else 'mort_hosp'
    id_col   = vital_train.index.names[0]
    print(f'Coluna de mortalidade: {mort_col}')
    print(f'Coluna de ID: {id_col}')

    def build_head(vital_df, inv_df):
        """
        Concatena vital + inv por paciente num único array (n_features, n_horas).
        vital: 184 features
        inv:   16 features
        total: 200 features — compatível com o código original
        Ordem das colunas inv após concatenação:
          índice 184: vent         <- usado por filter_arf
          índice 185: antibiotic
          índice 186: dopamine     <- usado por filter_shock
          índice 187: epinephrine  <- usado por filter_shock
          índice 188: norepinephrine <- usado por filter_shock
          índice 189: phenylephrine  <- usado por filter_shock
          índice 190: vasopressin    <- usado por filter_shock
          índice 191: dobutamine
          ...
        """
        head = []
        vital_grouped = dict(list(vital_df.groupby(level=id_col)))
        inv_grouped   = dict(list(inv_df.groupby(level=id_col)))

        for stay_id, vital_group in vital_grouped.items():
            vital_arr = vital_group.values.T.astype(np.float32)  # (184, n_horas)

            if stay_id in inv_grouped:
                inv_arr = inv_grouped[stay_id].values.T.astype(np.float32)  # (16, n_horas)
            else:
                # se não houver registo de intervenção preenche com zeros
                inv_arr = np.zeros((inv_df.shape[1], vital_arr.shape[1]), dtype=np.float32)

            # garantir que as duas tabelas têm o mesmo número de horas
            n_horas = min(vital_arr.shape[1], inv_arr.shape[1])
            combined = np.concatenate([vital_arr[:, :n_horas],
                                       inv_arr[:, :n_horas]], axis=0)  # (200, n_horas)
            head.append(combined)

        return head

    def build_static(static_df, mort_col):
        """
        Guarda apenas a coluna de mortalidade como array [valor] por paciente.
        A coluna 0 é sempre mortalidade — necessário para filter_los e task 0.
        """
        result = []
        for stay_id, row in static_df.iterrows():
            mort_val = float(row[mort_col]) if not pd.isna(row[mort_col]) else 0.0
            result.append(np.array([mort_val], dtype=np.float32))
        return result

    print('A construir train_head (vital + inv)...')
    train_head = build_head(vital_train, inv_train)
    print('A construir dev_head...')
    dev_head   = build_head(vital_dev, inv_dev)
    print('A construir test_head...')
    test_head  = build_head(vital_test, inv_test)

    print('A construir static filters...')
    static_train_filter = build_static(static_train, mort_col)
    static_dev_filter   = build_static(static_dev, mort_col)
    static_test_filter  = build_static(static_test, mort_col)

    # verificação
    print(f'Shape do primeiro paciente train: {train_head[0].shape}')
    print(f'Deve ser (200, n_horas)')
    print(f'Static do primeiro paciente: {static_train_filter[0]}')

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