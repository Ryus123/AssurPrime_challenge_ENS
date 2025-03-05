#############################################################
#### Import
#############################################################
import pandas as pd
import numpy as np
from category_encoders import CountEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split


# Charger les données
print("Chargement des données...")
X = pd.read_csv("data/train_input.csv")
y = pd.read_csv("data/train_output.csv")


#############################################################
#### Traitement des données
#############################################################
print("Données chargées avec succès.")


# Traitement des valeurs manquantes dans les colonnes numériques
print("Traitement des valeurs manquantes dans les colonnes numériques...")
numeric_columns = X.select_dtypes(include=['number']).columns

# Remplir les NaN avec 0 pour les colonnes numériques
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Identifier les colonnes non numériques
fill_cols = [item for item in X.columns if item not in numeric_columns]

# Remplir les NaN des colonnes non numériques avec une valeur par défaut (-999)
X[fill_cols] = X[fill_cols].fillna(-999)

print("Traitement des valeurs manquantes terminé.")

# Préparation des données pour l'entraînement
print("Préparation des données pour l'entraînement...")



# Split the validation and train set
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Suppression des colonnes inutiles
# X_train = X_train.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
X_train = X.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
y_train = y






# Encodage des variables catégoriques avec CountEncoder
encoder = CountEncoder(cols=fill_cols)
encoder.fit(X_train)
X_train_enc = encoder.transform(X_train)

print("Préparation terminée.")


#############################################################
#### Entraîner les modèles
#############################################################
print("Entraînement des modèles")

# Standardiser les données
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)

# Appliquer l'ACP avec conservation de 95% de la variance
pca = PCA(n_components=0.4)  # Conserver 95% de la variance
X_train_enc = pca.fit_transform(X_train_enc)



# Modèle pour prédire 'FREQ' avec une loi de Poisson
glm_freq = PoissonRegressor(alpha=.6)  # Régularisation faible
glm_freq.fit(X_train_enc, y_train['FREQ'])
print("Modèle GLM Poisson pour 'FREQ' entraîné avec succès.")

# Prédire 'FREQ' sur l'ensemble d'entraînement
y_train_pred_freq = glm_freq.predict(X_train_enc)

# Modèle pour prédire 'CM' avec une loi de Tweedie (power=1.5)
# Filtrer les données pour éviter les valeurs invalides
mask = y_train['CM'] > 0  # Garder uniquement les valeurs strictement positives

# Entraîner le GLM Tweedie uniquement sur ces données
glm_cm = TweedieRegressor(power=1.35, alpha=1e-8, max_iter=1000)
glm_cm.fit(X_train_enc[mask], y_train['CM'][mask])

print("Modèle GLM Tweedie pour 'CM' entraîné avec succès.")

# Prédire 'CM' (remettre les valeurs à 0 pour les cas exclus)
y_train_pred_cm = glm_cm.predict(X_train_enc)
y_train_pred_cm[~mask] = 0  # Remettre 0 pour les valeurs initialement nulles

# Calculer la prédiction combinée pour 'CHARGE'
y_train_pred = y_train_pred_freq * y_train_pred_cm * y_train['ANNEE_ASSURANCE']

# Calculer le RMSE sur l'ensemble d'entraînement
rmse = np.sqrt(mean_squared_error(y_train['CHARGE'], y_train_pred))

print(f"RMSE sur l'ensemble d'entraînement : {rmse:.2f}")




#############################################################
#### Validation
#############################################################

# print("Validation des modèles...")
# X_val_enc = encoder.transform(X_val.drop(['ID', 'ANNEE_ASSURANCE'], axis=1))
# X_val_enc = scaler.transform(X_val_enc)
# X_val_enc = pca.transform(X_val_enc)

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
# rmse = np.sqrt(mean_squared_error(y_val['CHARGE'], y_val_pred['CHARGE']))

# print(f"RMSE sur l'ensemble de validation : {rmse:.2f}")



#############################################################""
#### Test
#############################################################""


# Traitement des données de test
X_test = pd.read_csv("data/test_input.csv")
print("Traitement des données de test...")

# Remplir les valeurs manquantes
# Remplir les NaN avec 0 pour les colonnes numériques
X_test[numeric_columns] = X_test[numeric_columns].fillna(X_test[numeric_columns].mean())
# Identifier les colonnes non numériques
fill_cols = [item for item in X.columns if item not in numeric_columns]
# Remplir les NaN des colonnes non numériques avec une valeur par défaut (-999)
X_test[fill_cols] = X_test[fill_cols].fillna(-999)

# Suppression des colonnes inutiles

X_test_model = encoder.transform(X_test.drop(['ID', 'ANNEE_ASSURANCE'], axis=1))
X_test_model = scaler.transform(X_test_model)
X_test_model = pca.transform(X_test_model)

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