import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# CORS ultra-abierto para evitar bloqueos en Google AI Studio
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CARGA DE RECURSOS ---
try:
    master_df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
    m_attend = joblib.load('model_attendance.joblib')
    print("✅ Conexión con CSV y Modelos establecida")
except Exception as e:
    print(f"❌ Error crítico de carga: {e}")
    master_df = pd.DataFrame()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard_kpis():
    try:
        # Limpieza de datos: Convertir a número y llenar vacíos con 0
        df = master_df.copy()
        df['ENTRAN'] = pd.to_numeric(df['ENTRAN'], errors='coerce').fillna(0)
        df['APUNTADOS'] = pd.to_numeric(df['APUNTADOS'], errors='coerce').fillna(0)
        df['VALOR_TICKET'] = pd.to_numeric(df['VALOR_TICKET'], errors='coerce').fillna(0)
        
        total_entran = int(df['ENTRAN'].sum())
        total_apuntados = int(df['APUNTADOS'].sum())
        
        # PREDICCIÓN REAL: Usamos el total de registrados actuales como entrada al modelo
        # El modelo Random Forest espera [[valor]]
        if not m_attend:
            pred_val = total_entran # Fallback si el modelo no carga
        else:
            pred_val = m_attend.predict(np.array([[total_apuntados]]))[0]
        
        # Lógica de cordura: La predicción no puede ser menor a los que ya entraron
        asistencia_final = max(int(pred_val), total_entran)
        
        tasa = (total_entran / total_apuntados * 100) if total_apuntados > 0 else 0

        return jsonify({
            "total_invitados": total_entran,
            "total_registrados": total_apuntados,
            "tasa_conversion": f"{tasa:.2f}%",
            "prediccion_asistencia": asistencia_final,
            "ticket_promedio": int(df['VALOR_TICKET'].mean())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
