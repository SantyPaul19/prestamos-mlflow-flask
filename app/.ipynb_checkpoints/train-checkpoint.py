import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# Cargar datos desde SQLite utilizando la direccion de mi base de datos
conn = sqlite3.connect('..//data//prestamos.db')
df = pd.read_sql_query("SELECT * FROM prestamos", conn)
conn.close()

# Limpiar datos
df = df.dropna()

# Reemplazar '3+' en columna Dependents y convertir a entero, ya que despues nos
# puede dar un error
if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].astype(int)

# Codificar columnas categóricas
le = LabelEncoder()
cols_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols_to_encode:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Separar features y target
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Dividir datos para el onjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar MLflow en el puerto 9090
mlflow.set_tracking_uri("http://127.0.0.1:9090")
# Configurar MLflow el nombre del experimento
mlflow.set_experiment("prestamos_rf")

with mlflow.start_run():
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Registrar en MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="model")
    # Crea un modelo local
    joblib.dump(model, "..//model//modelo_local.pkl")

    print(f"✅ Entrenamiento completado. Accuracy: {acc:.4f}")
