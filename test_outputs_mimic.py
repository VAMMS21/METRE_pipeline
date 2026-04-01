import pandas as pd
import os

# Substituir pelo teu caminho real para os ficheiros gerados
output_dir = "/Users/vascoantoniosilva/Desktop/ICU_Pipeline/METRE_pipeline/output"

# Abrir as estatísticas de normalização
mean_std_file = os.path.join(output_dir, "MIMIC_mean_std_stats.h5")
mean_std = pd.read_hdf(mean_std_file, key='MIMIC_mean_std')
print("Exemplo das estatísticas de normalização:")
print(mean_std.head())

# Abrir os dados divididos em treino, validação e teste
split_file = os.path.join(output_dir, "MIMIC_split.hdf5")

vital_train = pd.read_hdf(split_file, key='vital_train')
static_train = pd.read_hdf(split_file, key='static_train')

print("\nExemplo dos sinais vitais (vital_train):")
print(vital_train.head())

print("\nExemplo dos dados estáticos (static_train):")
print(static_train.head())