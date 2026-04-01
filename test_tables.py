import pandas as pd
import h5py

keys = ['vital_train', 'inv_train', 'static_train']
datasets = {
    'MIMIC': './output/MIMIC_split.hdf5',
    'eICU': './output/eICU_split.hdf5'
}

for db_name, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {db_name}")
    print(f"{'='*60}")
    for key in keys:
        df = pd.read_hdf(path, key=key)
        print(f"\n--- {key} ---")
        print(f"Shape: {df.shape}")
        print(f"Index: {df.index.names}")
        print(f"Colunas ({len(df.columns)}):")
        for col in df.columns.tolist():
            print(f"  {col}")