import pandas as pd

# Charger le fichier CSV
X = pd.read_csv("data/train_input.csv" )

# Seuil : 50% de colonnes doivent être NaN
seuil = 3*X.shape[1] / 4

# Nombre de lignes contenant plus de 50% de NaN
nb_lignes_nan = (X.isna().sum(axis=1) > seuil).sum()

# Proportion
proportion = nb_lignes_nan / X.shape[0]

print(f"Proportion de lignes avec plus de 50% de NaN : {proportion:.2%}")