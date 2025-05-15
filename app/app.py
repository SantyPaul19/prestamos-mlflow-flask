from flask import Flask, request, render_template
import mlflow
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")
# Configurar MLflow e nombre de el experimento que creamos en train.py
EXPERIMENT_NAME = "prestamos_rf"

# Obtener automáticamente el último modelo exitoso
def load_latest_model():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No se encontró el experimento '{EXPERIMENT_NAME}'")

    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        filter_string="attributes.status = 'FINISHED'"
    )

    if runs.empty:
        raise ValueError("No hay ejecuciones exitosas en MLflow")

    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"mlruns/{experiment_id}/{latest_run_id}/artifacts/model"
    return mlflow.sklearn.load_model(model_uri)
    print(model_uri)

# Cargar el modelo una vez al iniciar la app, se carga desde MlFlow un modelo registrado y si 
# no encuentra o no se puede conectar toma el local creado
try:
    model = load_latest_model()
except:
    import joblib
    model = joblib.load("..//model//modelo_local.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            'Gender': int(request.form['Gender']),
            'Married': int(request.form['Married']),
            'Dependents': int(request.form['Dependents']),
            'Education': int(request.form['Education']),
            'Self_Employed': int(request.form['Self_Employed']),
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': int(request.form['Property_Area'])
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        resultado = "✅ Aprobado" if prediction == 1 else "❌ Rechazado"
        return render_template('index.html', prediction_text=f"Resultado del préstamo: {resultado}")

if __name__ == '__main__':
    app.run(debug=True)
