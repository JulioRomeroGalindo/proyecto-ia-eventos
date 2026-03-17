import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# Configuración de CORS abierta para evitar el bloqueo en AI Studio
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CARGA DE DATOS Y MODELOS ---
try:
    # Carga del CSV
    master_df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
    
    # Carga de modelos (asegúrate de que estos archivos estén en tu repo de GitHub)
    m_attend = joblib.load('model_attendance.joblib')
    # m_profit = joblib.load('model_profitability.joblib') # Opcional si lo usas luego
    # m_nn = tf.keras.models.load_model('model_revenue.h5', compile=False) # Opcional
    
    print("✅ Recursos cargados exitosamente")
except Exception as e:
    print(f"❌ Error al cargar archivos: {e}")
    # Creamos un DF vacío por si falla la carga para que el server no crashee
    master_df = pd.DataFrame()

@app.route('/')
def health_check():
    return jsonify({"status": "online", "message": "Servidor de Eventos listo"})

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard_kpis():
    try:
        # Cálculos de lógica de negocio
        total_entran = int(master_df['ENTRAN'].sum())
        total_apuntados = int(master_df['APUNTADOS'].sum())
        conversion_rate = (total_entran / total_apuntados * 100) if total_apuntados > 0 else 0
        
        # Predicción base usando el promedio de apuntados para el dashboard inicial
        avg_apuntados = master_df['APUNTADOS'].mean()
        pred_base = m_attend.predict(np.array([[avg_apuntados]]))[0]

        # IMPORTANTE: Estos nombres de llaves coinciden con tu Dashboard de AI Studio
        return jsonify({
            "total_invitados": total_entran,
            "tasa_conversion": f"{conversion_rate:.2f}%",
            "prediccion_asistencia": int(pred_base),
            "total_registrados": total_apuntados,
            "ticket_promedio": int(master_df['VALOR_TICKET'].mean())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/predict/attendance', methods=['POST'])
def pred_attend():
    try:
        data = request.json
        apuntados = data.get('apuntados', 100) # valor por defecto
        prediction = m_attend.predict(np.array([[apuntados]]))
        return jsonify({"asistencia_estimada": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
