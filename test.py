import pandas as pd

# Charger le fichier CSV

colonnes_a_specifier = ['FRCH1', 'FRCH2', 'DEROG12', 'DEROG13', 'DEROG14', 
                       'RISK6', 'RISK8', 'RISK9', 'RISK12', 'RISK13', 
                       'EQUIPEMENT2', 'EQUIPEMENT5', 'ESPINSEE']

dtype_dict = {col: str for col in colonnes_a_specifier}

df = pd.read_csv("data/train_input.csv", dtype=dtype_dict, low_memory=False)


