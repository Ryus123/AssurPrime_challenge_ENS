#############################################################
#### Import
#############################################################
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import time


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
print(y.shape)
indices_a_supprimer2 = X[X["ANNEE_ASSURANCE"] < 0.01].index.tolist()
X = X.drop(index=indices_a_supprimer2)
y = y.drop(index=indices_a_supprimer2)
print(y.shape)


# Réinitialiser les index
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

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
# X_train = X_train.drop(['ID'], axis=1)

X_train = X
X_train = X_train.drop(['ID'], axis=1)
y_train = y

#############################################################
#### Reduction des données
#############################################################

X_train_enc = X_train[numeric_columns]

print('Applique ACP')
#Standardiser les données
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)

# Appliquer l'ACP avec conservation de 95% de la variance
pca = PCA(n_components=0.37)  # Conserver 95% de la variance
X_train_enc = pca.fit_transform(X_train_enc)

print("Préparation terminée.")


#############################################################
#### Entraînement
#############################################################
print("Entraînement des modèles XGBoost")
t_start = time.time()
# Modèle XGBoost pour prédire 'FREQ' (loi de Poisson)
xgb_freq = XGBRegressor(
    objective='count:poisson',
    subsample=0.6,
    reg_lambda=0.1,
    reg_alpha=10.0,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1388888888888889,
    colsample_bytree=0.8
)

xgb_freq.fit(X_train_enc, y_train['FREQ'])
print("Modèle XGBoost Poisson pour 'FREQ' entraîné avec succès.")

# Prédire 'FREQ' sur l'ensemble d'entraînement
y_train_pred_freq = xgb_freq.predict(X_train_enc)


# Filtrer les données pour éviter les valeurs invalides (CM doit être > 0)
mask = y_train['CM'] > 0  # Garder uniquement les valeurs strictement positives

# Modèle XGBoost pour prédire 'CM' (loi de Tweedie)
xgb_cm = XGBRegressor(
    objective='reg:tweedie',
    tweedie_variance_power=1.7375,
    subsample=0.6,
    reg_lambda=1.0,
    reg_alpha=0.1,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.07444444444444444,
    colsample_bytree=1.0
)

# Entraîner uniquement sur les valeurs positives
xgb_cm.fit(X_train_enc[mask], y_train['CM'][mask])
print("Modèle XGBoost Tweedie pour 'CM' entraîné avec succès.")
model_building_time = time.time() - t_start
# Prédire 'CM' (remettre les valeurs à 0 pour les cas exclus)
y_train_pred_cm = xgb_cm.predict(X_train_enc)
y_train_pred_cm[~mask] = 0  # Remettre 0 pour les valeurs initialement nulles

# Calculer la prédiction combinée pour 'CHARGE'
y_train_pred = y_train_pred_freq * y_train_pred_cm * y_train['ANNEE_ASSURANCE']

# Calculer le RMSE sur l'ensemble d'entraînement
rmse = np.sqrt(mean_squared_error(y_train['CHARGE'], y_train_pred))
rmse_freq = np.sqrt(mean_squared_error(y_train['FREQ'], y_train_pred_freq))
rmse_cm = np.sqrt(mean_squared_error(y_train['CM'], y_train_pred_cm))

print("------- Training score\n")
print(f"RMSE sur l'ensemble de validation : {rmse:.2f}")
print(f"RMSE - FREQ sur l'ensemble de validation : {rmse_freq/y_train['FREQ'].std():.4f}")
print(f"RMSE - CM sur l'ensemble de validation : {rmse_cm/y_train['CM'].std():.4f}")

print(f'Model computation time = {model_building_time:.2f}s')
#############################################################
#### Validation
#############################################################

# print("Validation des modèles...")
# # X_val_enc = encoder.transform(X_val.drop(['ID'], axis=1))

# X_val_enc = X_val.drop(['ID'], axis=1)
# X_val_enc = X_val_enc[numeric_columns]

# X_val_enc = scaler.transform(X_val_enc)
# X_val_enc = pca.transform(X_val_enc)

# print("Prédictions sur l'ensemble de validation...")
# # Prédire 'FREQ' et 'CM' sur les données de validation
# y_val_pred_freq = xgb_freq.predict(X_val_enc)
# y_val_pred_cm = xgb_cm.predict(X_val_enc)

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
# print(f"RMSE - FREQ sur l'ensemble de validation : {rmse_val_freq/y_val['FREQ'].std():.4f}")
# print(f"RMSE - CM sur l'ensemble de validation : {rmse_val_cm/y_val['CM'].std():.4f}")


#############################################################""
#### Test
#############################################################""


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

# X_test_model = encoder.transform(X_test.drop(['ID'], axis=1))

X_test_model = X_test.drop(['ID'], axis=1)
X_test_model = X_test_model[numeric_columns]
X_test_model = scaler.transform(X_test_model)
X_test_model = pca.transform(X_test_model)

print("Prédictions sur l'ensemble de validation...")
# Prédire 'FREQ' et 'CM' sur les données de validation
y_pred_freq = xgb_freq.predict(X_test_model)
y_pred_cm = xgb_cm.predict(X_test_model)
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