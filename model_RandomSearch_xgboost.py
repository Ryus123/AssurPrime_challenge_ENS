#############################################################
#### Import
#############################################################
import pandas as pd
import numpy as np
from category_encoders import CountEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import numpy as np

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
indices_a_supprimer2 = X[X["ANNEE_ASSURANCE"] < 0.1].index.tolist()
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)
X_train = X_train.drop(['ID'], axis=1)

# X_train = X
# X_train = X_train.drop(['ID', 'ANNEE_ASSURANCE'], axis=1)
# y_train = y

#############################################################
#### Reduction des données
#############################################################

X_train_enc = X_train[numeric_columns]

print('Applique ACP')
#Standardiser les données
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)

# Appliquer l'ACP avec conservation de 95% de la variance
pca = PCA(n_components=0.4)  # Conserver 95% de la variance
X_train_enc = pca.fit_transform(X_train_enc)

print("Préparation terminée.")







#############################################################
#### Entraîner les modèles avec Random Search
#############################################################
print("Optimisation et entraînement du modèle XGBoost pour 'FREQ'")

# Définition du modèle XGBoost
xgb_freq = XGBRegressor(objective='count:poisson', eval_metric="rmse", random_state=42)

# Grille d'hyperparamètres pour Random Search
param_dist = {
    "n_estimators": [100, 300, 500],
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "max_depth": [3, 5, 7],
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "reg_alpha": np.logspace(-3, 1, 5),  # L1 regularization
    "reg_lambda": np.logspace(-3, 1, 5)  # L2 regularization
}

# Random Search avec validation croisée
random_search = RandomizedSearchCV(
    estimator=xgb_freq,
    param_distributions=param_dist,
    n_iter=20,  # Nombre d'itérations à tester
    cv=3,  # Validation croisée 3-fold
    scoring="neg_mean_squared_error",  # Critère d'évaluation
    verbose=2,
    n_jobs=-1
)

# Entraînement du modèle avec Random Search
random_search.fit(X_train_enc, y_train['FREQ'])

# Meilleur modèle obtenu
best_xgb_freq = random_search.best_estimator_
print("Meilleur modèle XGBoost pour 'FREQ' entraîné avec succès.")

# Prédictions sur l'ensemble d'entraînement
y_train_pred_freq = best_xgb_freq.predict(X_train_enc)

print("\n\nMeilleurs hyperparamètres pour 'FREQ' :", random_search.best_params_)





#############################################################
#### Modèle XGBoost pour 'CM' avec Tweedie
#############################################################
print("Optimisation et entraînement du modèle XGBoost pour 'CM'")

# Filtrer les données pour éviter les valeurs invalides
mask = y_train['CM'] > 0  # Garder uniquement les valeurs strictement positives

# Définition du modèle XGBoost pour CM
xgb_cm = XGBRegressor(objective="reg:tweedie", eval_metric="rmse", random_state=42)

# Grille d'hyperparamètres pour Random Search (CM)
param_dist_cm = {
    "n_estimators": [100, 300, 500],
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "max_depth": [3, 5, 7 ],
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "reg_alpha": np.logspace(-3, 1, 5),
    "reg_lambda": np.logspace(-3, 1, 5),
    "tweedie_variance_power": np.linspace(1.1, 1.95, 5)  # Paramètre spécifique à Tweedie
}

# Random Search avec validation croisée pour CM
random_search_cm = RandomizedSearchCV(
    estimator=xgb_cm,
    param_distributions=param_dist_cm,
    n_iter=20,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=2,
    n_jobs=-1
)

# Entraînement du modèle XGBoost pour CM (seulement sur les valeurs valides)
random_search_cm.fit(X_train_enc[mask], y_train['CM'][mask])
best_xgb_cm = random_search_cm.best_estimator_
print("Meilleur modèle XGBoost pour 'CM' entraîné avec succès.")

# Prédictions pour CM
y_train_pred_cm = best_xgb_cm.predict(X_train_enc)

print("\n\nMeilleurs hyperparamètres pour 'CM' :", random_search_cm.best_params_)





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

print("Validation des modèles...")
# X_val_enc = encoder.transform(X_val.drop(['ID'], axis=1))

X_val_enc = X_val.drop(['ID'], axis=1)
X_val_enc = X_val_enc[numeric_columns]

X_val_enc = scaler.transform(X_val_enc)
X_val_enc = pca.transform(X_val_enc)

print("Prédictions sur l'ensemble de validation...")
# Prédire 'FREQ' et 'CM' sur les données de validation
y_val_pred_freq = best_xgb_freq.predict(X_val_enc)
y_val_pred_cm = best_xgb_cm.predict(X_val_enc)

# Combiner les prédictions
y_val_pred = pd.concat([
    X_val[['ID']].reset_index(drop=True),
    pd.DataFrame(y_val_pred_freq, columns=['FREQ']),
    pd.DataFrame(y_val_pred_cm, columns=['CM']),
    X_val[['ANNEE_ASSURANCE']].reset_index(drop=True)
], axis=1)

# Calculer la prédiction combinée pour 'CHARGE'
y_val_pred['CHARGE'] = y_val_pred['FREQ'] * y_val_pred['CM'] * y_val_pred['ANNEE_ASSURANCE']

# Calculer le RMSE sur l'ensemble d'entraînement
rmse_val = np.sqrt(mean_squared_error(y_val['CHARGE'], y_val_pred['CHARGE']))
rmse_val_freq = np.sqrt(mean_squared_error(y_val['FREQ'], y_val_pred['FREQ']))
rmse_val_cm = np.sqrt(mean_squared_error(y_val['CM'], y_val_pred['CM']))

print("------- Validation score\n")
print(f"RMSE sur l'ensemble de validation : {rmse_val:.2f}")
print(f"RMSE - FREQ sur l'ensemble de validation : {rmse_val_freq/y_val['FREQ'].mean():.4f}")
print(f"RMSE - CM sur l'ensemble de validation : {rmse_val_cm/y_val['CM'].mean():.4f}")

print("\n\nMeilleurs hyperparamètres pour 'FREQ' :", random_search.best_params_)
print("\n\nMeilleurs hyperparamètres pour 'CM' :", random_search_cm.best_params_)

# Meilleurs hyperparamètres pour 'FREQ' : {'subsample': 0.6, 'reg_lambda': 0.1, 'reg_alpha': 10.0, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1388888888888889, 'colsample_bytree': 0.8}


# Meilleurs hyperparamètres pour 'CM' : {'tweedie_variance_power': 1.7375, 'subsample': 0.6, 'reg_lambda': 1.0, 'reg_alpha': 0.1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.07444444444444444, 'colsample_bytree': 1.0}     
