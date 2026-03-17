import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras  # Importamos keras directamente
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def cargar_recursos():
    base_path = os.path.dirname(os.path.abspath(__file__))
    nombres = {
        "asistencia": "model_attendance_v2.joblib",
        "rentabilidad": "model_profitability_v2.joblib",
        "segmentos": "model_segments_v2.joblib",
        "revenue": "model_revenue_v2.h5"
    }
    recursos = {}
    
    try:
        # Carga de Scikit-Learn
        recursos["asistencia"] = joblib.load(os.path.join(base_path, nombres["asistencia"]))
        recursos["rentabilidad"] = joblib.load(os.path.join(base_path, nombres["rentabilidad"]))
        recursos["segmentos"] = joblib.load(os.path.join(base_path, nombres["segmentos"]))
        
        # SOLUCIÓN DEFINITIVA: Usar el cargador nativo de Keras 3
        ruta_h5 = os.path.join(base_path, nombres["revenue"])
        recursos["revenue"] = keras.models.load_model(ruta_h5, compile=False)
        
        print("✅ Modelos cargados con el motor nativo de Keras 3")
        return recursos, None
    except Exception as e:
        print(f"❌ Error en carga: {str(e)}")
        return None, str(e)

MODELS, ERROR_MSG = cargar_recursos()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    global MODELS, ERROR_MSG
    if MODELS is None:
        MODELS, ERROR_MSG = cargar_recursos()
        if MODELS is None:
            return jsonify({"error": f"Modelos no listos: {ERROR_MSG}"}), 500

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "REPORTE_MAESTRO_DEFINITIVO.csv")
        df = pd.read_csv(csv_path, sep=';', encoding='utf-16')
        
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float(df['COSTO_TOTAL'].mean())

        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        
        # Inferencia
        prob_att = MODELS["asistencia"].predict_proba(test_df)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)
        res_prof = MODELS["rentabilidad"].predict(test_df)[0]
        res_seg = MODELS["segmentos"].predict(test_df.values)[0]
        
        # Predicción Revenue con input formateado para Keras 3
        input_nn = np.array([[t_apuntados, avg_costo]], dtype="float32")
        pred_rev = float(MODELS["revenue"](input_nn, training=False).numpy()[0][0])

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
        return jsonify({"error": f"Error en ejecución: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
