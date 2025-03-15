import pandas as pd

# Charger le fichier CSV
X = pd.read_csv("data/train_input.csv" )

print(X.loc[:, X.isna().mean() * 100 > 35])