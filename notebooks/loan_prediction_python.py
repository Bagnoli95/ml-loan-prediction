# para no mostrar mensajes en consola de los warnings
import warnings
warnings.filterwarnings('ignore')
# Import inicial, exploracion e imputacion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Entrenamiento, evaluacion
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Cargar el dataset y hacer un resumen
df = pd.read_csv("../data/train.csv")
df.head()



# Estrategia: Rellenar valores categóricos con el modo y numéricos con la mediana
categorical_cols_with_nan = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
numerical_cols_with_nan = ['LoanAmount']

# Rellenar valores faltantes
for col in categorical_cols_with_nan:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_cols_with_nan:
    df[col].fillna(df[col].median(), inplace=True)

# Confirmar que no queden valores faltantes
missing_after_fill = df.isnull().sum()

# Detección de outliers usando el rango intercuartil (IQR)
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
outliers = {}

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

missing_after_fill, {col: len(outliers[col]) for col in numerical_cols}


# Transformación logarítmica para mitigar el impacto de outliers
for col in numerical_cols:
    df[col] = np.log1p(df[col])  # log1p(x) = log(1 + x)

# Codificación de variables categóricas con One-Hot Encoding
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
dataset_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Escalado de características numéricas
scaler = StandardScaler()
numerical_cols_transformed = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
dataset_encoded[numerical_cols_transformed] = scaler.fit_transform(dataset_encoded[numerical_cols_transformed])

# Dividir el conjunto de datos en entrenamiento y prueba
# Crear el dataset de entrenamiento y prueba
X = dataset_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = dataset_encoded['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Convertir Loan_Status a binario

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo LightGBM básico
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Predicciones y evaluación inicial
y_pred = lgb_model.predict(X_test)
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

print("Métricas del modelo inicial:")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisión: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Optimización de hiperparámetros con GridSearchCV
param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'boosting_type': ['gbdt'],
    'objective': ['binary'],
}

grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test)
y_pred_proba_optimized = best_model.predict_proba(X_test)[:, 1]

print("\nMétricas del modelo optimizado:")
print(f"Exactitud: {accuracy_score(y_test, y_pred_optimized):.4f}")
print(f"Precisión: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_optimized):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_optimized):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_optimized):.4f}")

# Visualización de importancia de características
feature_importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.title("Importancia de características")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.show()