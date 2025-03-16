#############################################################
#### Import
#############################################################
import pandas as pd
import numpy as np
from category_encoders import CountEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

import rfflearn.cpu as rfflearn

from sklearn.model_selection import train_test_split


# Charger les données
print("Chargement des données...")
X = pd.read_csv("data/train_input.csv")
y = pd.read_csv("data/train_output.csv")

colonnes_supprimees = X.columns[X.isna().mean() * 100 > 70]
X.drop(columns=colonnes_supprimees, inplace=True)

seuil = X.shape[1] / 2
indices_a_supprimer = X.index[X.isna().sum(axis=1) > seuil].tolist()
# Suppression des lignes avec plus de 50% de NaN
X = X.drop(index=indices_a_supprimer)
y = y.drop(index=indices_a_supprimer)
# Réinitialiser les index
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)


print(X.shape)  # (devrait afficher (N, nb_features))
print(y.shape)
#############################################################
#### Traitement des données
#############################################################
print("Données chargées avec succès.")


# Traitement des valeurs manquantes dans les colonnes numériques
print("Traitement des valeurs manquantes dans les colonnes numériques...")
# Suppression des colonnes inutiles
numeric_columns = X.drop(['ID', 'ANNEE_ASSURANCE'], axis=1).select_dtypes(include=['number']).columns
# Remplir les NaN avec 0 pour les colonnes numériques
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Identifier les colonnes non numériques
fill_cols = [item for item in X.columns if item not in numeric_columns and item not in ['ID', 'ANNEE_ASSURANCE']]

# Process
SURFACE = [chaine for chaine in fill_cols if chaine.startswith("SURFACE")]
X[SURFACE] = X[SURFACE].apply(pd.to_numeric, errors='coerce')
NBJTX = [chaine for chaine in fill_cols if chaine.startswith("NBJTX")]
X[NBJTX] = X[NBJTX].fillna('').astype(str).map(lambda x: x.split(" ")[-1])
X[NBJTX] = X[NBJTX].apply(pd.to_numeric, errors='coerce')
NBJRR = [chaine for chaine in fill_cols if chaine.startswith("NBJRR")]
X[NBJRR] = X[NBJRR].fillna('').astype(str).map(lambda x: x.split(" ")[-1])
X[NBJRR] = X[NBJRR].apply(pd.to_numeric, errors='coerce')



numeric_columns = X.drop(['ID', 'ANNEE_ASSURANCE'], axis=1).select_dtypes(include=['number']).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
# Identifier les colonnes non numériques
fill_cols = [item for item in X.columns if item not in numeric_columns and item not in ['ID', 'ANNEE_ASSURANCE']]




# Remplir les NaN des colonnes non numériques avec une valeur par défaut (-999)
X[fill_cols] = X[fill_cols].fillna(-999)

print("Traitement des valeurs manquantes terminé.")

# Préparation des données pour l'entraînement
print("Préparation des données pour l'entraînement...")

# Split the validation and train set
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)
# X_train = X_train.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)

X_train = X
X_train = X_train.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
y_train = y

#############################################################
#### Reduction des données
#############################################################

# Encodage des variables catégoriques avec CountEncoder
encoder = OrdinalEncoder(cols=fill_cols)
encoder.fit(X_train)
X_train_enc = encoder.transform(X_train)

print('Applique ACP')
#Standardiser les données
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)


print("Préparation terminée.")

#############################################################
#### Entraîner les modèles
#############################################################
print("Entraînement des modèles")

# Modèle pour prédire 'FREQ' avec une loi de Poisson
glm_freq = PoissonRegressor(alpha=.8, solver='newton-cholesky' )  # Régularisation faible
glm_freq.fit(X_train_enc, y_train['FREQ'])
print("Modèle GLM Poisson pour 'FREQ' entraîné avec succès.")

# Prédire 'FREQ' sur l'ensemble d'entraînement
y_train_pred_freq = glm_freq.predict(X_train_enc)

# Modèle pour prédire 'CM' avec une loi de Tweedie (power=1.5)
# Filtrer les données pour éviter les valeurs invalides


# Entraîner le GLM Tweedie uniquement sur ces données
glm_cm = rfflearn.RFFRegressor(dim_kernel=6000, std_kernel=1.)
glm_cm.fit(X_train_enc, y_train['CM'])


print("Modèle GLM Tweedie pour 'CM' entraîné avec succès.")

# Prédire 'CM' (remettre les valeurs à 0 pour les cas exclus)
y_train_pred_cm = glm_cm.predict(X_train_enc)

# Calculer la prédiction combinée pour 'CHARGE'
y_train_pred = y_train_pred_freq * y_train_pred_cm * y_train['ANNEE_ASSURANCE']

# Calculer le RMSE sur l'ensemble d'entraînement
rmse = np.sqrt(mean_squared_error(y_train['CHARGE'], y_train_pred))
rmse_freq = np.sqrt(mean_squared_error(y_train['FREQ'], y_train_pred_freq))
rmse_cm = np.sqrt(mean_squared_error(y_train['CM'], y_train_pred_cm))

print("------- Training score\n")
print(f"RMSE sur l'ensemble de validation : {rmse:.2f}")
print(f"RMSE - FREQ sur l'ensemble de validation : {rmse_freq/y_train['FREQ'].mean():.4f}")
print(f"RMSE - CM sur l'ensemble de validation : {rmse_cm/y_train['CM'].mean():.4f}")




#############################################################
#### Validation
#############################################################

# print("Validation des modèles...")
# X_val_enc = encoder.transform(X_val.drop(['ID', 'ANNEE_ASSURANCE'], axis=1))
# X_val_enc = scaler.transform(X_val_enc)

# print("Prédictions sur l'ensemble de validation...")
# # Prédire 'FREQ' et 'CM' sur les données de validation
# y_val_pred_freq = glm_freq.predict(X_val_enc)
# y_val_pred_cm = glm_cm.predict(X_val_enc)

# # Combiner les prédictions
# y_val_pred = pd.concat([
#     X_val[['ID']].reset_index(drop=True),
#     pd.DataFrame(y_val_pred_freq, columns=['FREQ']),
#     pd.DataFrame(y_val_pred_cm, columns=['CM']),
#     X_val[['ANNEE_ASSURANCE']].reset_index(drop=True)
# ], axis=1)

# # Calculer la prédiction combinée pour 'CHARGE'
# y_val_pred['CHARGE'] = y_val_pred['FREQ'] * y_val_pred['CM'] * y_val_pred['ANNEE_ASSURANCE']

# # Calculer le RMSE sur l'ensemble d'entraînement
# rmse_val = np.sqrt(mean_squared_error(y_val['CHARGE'], y_val_pred['CHARGE']))
# rmse_val_freq = np.sqrt(mean_squared_error(y_val['FREQ'], y_val_pred['FREQ']))
# rmse_val_cm = np.sqrt(mean_squared_error(y_val['CM'], y_val_pred['CM']))

# print("------- Validation score\n")
# print(f"RMSE sur l'ensemble de validation : {rmse_val:.2f}")
# print(f"RMSE - FREQ sur l'ensemble de validation : {rmse_val_freq/y_val['FREQ'].mean():.4f}")
# print(f"RMSE - CM sur l'ensemble de validation : {rmse_val_cm/y_val['CM'].mean():.4f}")



# ############################################################""
# ### Test
# ############################################################""


# Traitement des données de test
X_test = pd.read_csv("data/test_input.csv")
X_test.drop(columns=colonnes_supprimees, inplace=True)
print("Traitement des données de test...")

# Remplir les valeurs manquantes
numeric_columns = X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1).select_dtypes(include=['number']).columns
# Identifier les colonnes non numériques
fill_cols = [item for item in X.columns if item not in numeric_columns]



# Process
SURFACE = [chaine for chaine in fill_cols if chaine.startswith("SURFACE")]
X_test[SURFACE] = X_test[SURFACE].apply(pd.to_numeric, errors='coerce')
NBJTX = [chaine for chaine in fill_cols if chaine.startswith("NBJTX")]
X_test[NBJTX] = X_test[NBJTX].fillna('').astype(str).map(lambda x: x.split(" ")[-1])
X_test[NBJTX] = X_test[NBJTX].apply(pd.to_numeric, errors='coerce')
NBJRR = [chaine for chaine in fill_cols if chaine.startswith("NBJRR")]
X_test[NBJRR] = X_test[NBJRR].fillna('').astype(str).map(lambda x: x.split(" ")[-1])
X_test[NBJRR] = X_test[NBJRR].apply(pd.to_numeric, errors='coerce')

numeric_columns = X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1).select_dtypes(include=['number']).columns
X_test[numeric_columns] = X_test[numeric_columns].fillna(X_test[numeric_columns].mean())
# Identifier les colonnes non numériques
fill_cols = [item for item in X_test.columns if item not in numeric_columns and item not in ['ID', 'ANNEE_ASSURANCE']]




# Remplir les NaN des colonnes non numériques avec une valeur par défaut (-999)
X_test[fill_cols] = X_test[fill_cols].fillna(-999)

# Suppression des colonnes inutiles

X_test_model = encoder.transform(X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1))
X_test_model = scaler.transform(X_test_model)

print("Prédictions sur l'ensemble de validation...")
# Prédire 'FREQ' et 'CM' sur les données de validation
y_pred_freq = glm_freq.predict(X_test_model)
y_pred_cm = glm_cm.predict(X_test_model)
print("Prédictions terminées.")


# Exporter les prédictions dans un fichier CSV
print("Exportation des résultats...")
# Combiner les prédictions
y_pred = pd.concat([
    X_test[['ID']].reset_index(drop=True),
    pd.DataFrame(y_pred_freq, columns=['FREQ']),
    pd.DataFrame(y_pred_cm, columns=['CM']),
    X_test[['ANNEE_ASSURANCE']].reset_index(drop=True)
], axis=1)


# Calculer la prédiction combinée pour 'CHARGE'
y_pred['CHARGE'] = y_pred['FREQ'] * y_pred['CM'] * y_pred['ANNEE_ASSURANCE']
y_pred.to_csv('submission.csv', index=False)
print("Fichier de soumission créé : 'submission.csv'")