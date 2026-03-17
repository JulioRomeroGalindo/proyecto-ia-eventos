import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cargar_modelos():
    modelos = {}
    nombres = {
        "asistencia": "model_attendance_v2.joblib",
        "rentabilidad": "model_profitability_v2.joblib",
        "segmentos": "model_segments_v2.joblib",
        "revenue": "model_revenue_v2.h5"
    }
    
    try:
        modelos["asistencia"] = joblib.load(os.path.join(BASE_DIR, nombres["asistencia"]))
        modelos["rentabilidad"] = joblib.load(os.path.join(BASE_DIR, nombres["rentabilidad"]))
        modelos["segmentos"] = joblib.load(os.path.join(BASE_DIR, nombres["segmentos"]))
        modelos["revenue"] = tf.keras.models.load_model(os.path.join(BASE_DIR, nombres["revenue"]), compile=False)
        print("✅ TODOS LOS MODELOS CARGADOS DESDE:", BASE_DIR)
        return modelos
    except Exception as e:
        print(f"❌ ERROR CARGANDO MODELOS: {e}")
        # Esto imprimirá en Render qué archivos SI existen para comparar
        print("Archivos detectados:", os.listdir(BASE_DIR))
        return None

# Intentar cargar al iniciar
MODELS = cargar_modelos()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    try:
        if MODELS is None:
            return jsonify({"error": "Los modelos no están listos en el servidor"}), 500

        # Lectura de datos
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float(df['COSTO_TOTAL'].mean())

        # Predicciones
        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        
        # 1. Asistencia
        prob_att = MODELS["asistencia"].predict_proba(test_df)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)
        
        # 2. Rentabilidad y Segmentos
        res_prof = MODELS["rentabilidad"].predict(test_df)[0]
        res_seg = MODELS["segmentos"].predict(test_df.values)[0]
        
        # 3. Revenue
        test_rev = np.array([[t_apuntados, avg_costo]])
        pred_rev = float(MODELS["revenue"].predict(test_rev, verbose=0)[0][0])

        return jsonify({
            "asistentes_reales": t_entran,
            "registrados": t_apuntados,
            "conversion": f"{(t_entran/t_apuntados*100):.2f}%" if t_apuntados > 0 else "0%",
            "pred_asistencia": pred_asistencia,
            "ticket_prom": int(avg_costo),
            "rentabilidad": "Optima" if res_prof == 1 else "Baja",
            "perfil": "VIP" if res_seg == 1 else "Estandar",
            "revenue": f"${pred_rev:,.2f}"
        })
    except Exception as e:
        return jsonify({"error": f"Fallo en proceso: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
