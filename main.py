import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CARGA DE MODELOS CON RUTA ABSOLUTA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    try:
        m1 = joblib.load(os.path.join(BASE_DIR, 'model_attendance_v2.joblib'))
        m2 = joblib.load(os.path.join(BASE_DIR, 'model_profitability_v2.joblib'))
        m3 = joblib.load(os.path.join(BASE_DIR, 'model_segments_v2.joblib'))
        m4 = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model_revenue_v2.h5'), compile=False)
        return m1, m2, m3, m4
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODELS: {e}")
        return None, None, None, None

m_attend, m_profit, m_segments, m_revenue = load_models()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    try:
        # Carga de datos
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float(df['COSTO_TOTAL'].mean())

        # Si los modelos no cargaron, enviamos valores por defecto pero NO undefined
        if m_attend is None:
            return jsonify({"error": "Modelos no cargados en servidor"}), 500

        # Predicciones
        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        prob_att = m_attend.predict_proba(test_df)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)
        
        res_prof = m_profit.predict(test_df)[0]
        res_seg = m_segments.predict(test_df.values)[0]
        
        test_rev = np.array([[t_apuntados, avg_costo]])
        pred_rev = float(m_revenue.predict(test_rev, verbose=0)[0][0])

        # LLAVES SIMPLIFICADAS
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
