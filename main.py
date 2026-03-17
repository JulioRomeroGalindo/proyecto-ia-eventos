import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CARGA DE DATOS Y MODELOS ---
try:
    # Usamos el nombre exacto de tu archivo
    master_df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
    
    # IMPORTANTE: Estos nombres deben ser los mismos que descargaste de Colab
    m_attend = joblib.load('model_attendance.joblib')
    m_profit = joblib.load('model_profitability.joblib')
    m_nn = tf.keras.models.load_model('model_revenue.h5', compile=False)
    
    print("✅ Recursos cargados exitosamente")
except Exception as e:
    print(f"❌ Error al cargar archivos: {e}")

@app.route('/')
def health_check():
    return jsonify({"status": "online", "message": "Servidor de Eventos en Render listo"})

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard_kpis():
    try:
        total_entran = int(master_df['ENTRAN'].sum())
        total_apuntados = int(master_df['APUNTADOS'].sum())
        conversion_rate = (total_entran / total_apuntados * 100) if total_apuntados > 0 else 0
        
        # Ajustado a tu columna 'TIPO_COBRO'
        pagan_penalidad = master_df[master_df['TIPO_COBRO'].str.contains('Penalidad', na=False, case=False)].shape[0]
        penalty_ratio = (pagan_penalidad / len(master_df) * 100) if len(master_df) > 0 else 0
        
        return jsonify({
            "business_performance": {
                "conversion_rate": f"{conversion_rate:.2f}%",
                "penalty_ratio": f"{penalty_ratio:.2f}%",
                "total_guests": total_entran,
                "avg_ticket": int(master_df['VALOR_TICKET'].mean())
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/predict/attendance', methods=['POST'])
def pred_attend():
    try:
        data = request.json
        apuntados = data.get('apuntados', 0)
        # Tu modelo espera una matriz 2D [[valor]]
        prediction = m_attend.predict(np.array([[apuntados]]))
        return jsonify({"asistencia_estimada": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)