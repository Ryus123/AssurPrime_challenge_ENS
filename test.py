import pandas as pd

# Charger le fichier CSV
X = pd.read_csv("data/train_output.csv" )

print(X.shape)
print(X[X["ANNEE_ASSURANCE"] > 0.5].shape)