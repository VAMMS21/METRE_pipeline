import pandas as pd
import h5py

# ver estrutura
with h5py.File('./output/MIMIC_split.hdf5', 'r') as f:
    print("MIMIC keys:", list(f.keys()))

with h5py.File('./output/eICU_split.hdf5', 'r') as f:
    print("eICU keys:", list(f.keys()))

# carregar dados
mimic_vital = pd.read_hdf('./output/MIMIC_split.hdf5', key='vital_train')
eicu_vital = pd.read_hdf('./output/eICU_split.hdf5', key='vital_train')
mimic_static = pd.read_hdf('./output/MIMIC_split.hdf5', key='static_train')
eicu_static = pd.read_hdf('./output/eICU_split.hdf5', key='static_train')

# dimensões
print("MIMIC vital shape:", mimic_vital.shape)
print("eICU vital shape:", eicu_vital.shape)
print("MIMIC static shape:", mimic_static.shape)
print("eICU static shape:", eicu_static.shape)

# mortalidade
print("MIMIC mortalidade:", mimic_static['hosp_mort'].mean())
print("eICU mortalidade:", eicu_static['hosp_mort'].mean())