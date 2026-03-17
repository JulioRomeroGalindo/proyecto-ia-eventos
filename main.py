import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- RASTREADOR DE MODELOS ---
def load_models():
    # Nombres exactos que deben estar en GitHub
    files = {
        "attend": "model_attendance_v2.joblib",
        "profit": "model_profitability_v2.joblib",
        "segments": "model_segments_v2.joblib",
        "revenue": "model_revenue_v2.h5"
    }
    
    loaded = {}
    for key, name in files.items():
        if os.path.exists(name):
            try:
                if name.endswith('.h5'):
                    loaded[key] = tf.keras.models.load_model(name, compile=False)
                else:
                    loaded[key] = joblib.load(name)
                print(f"✅ Cargado: {name}")
            except Exception as e:
                print(f"❌ Error cargando {name}: {e}")
                loaded[key] = None
        else:
            print(f"⚠️ Archivo NO ENCONTRADO: {name}")
            loaded[key] = None
    return loaded

models = load_models()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    try:
        # 1. Verificar si todos los modelos están presentes
        if None in models.values():
            missing = [k for k, v in models.items() if v is None]
            return jsonify({"error": f"Modelos faltantes en el servidor: {missing}"}), 500

        # 2. Leer CSV
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float(df['COSTO_TOTAL'].mean())

        # 3. Predicciones
        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        
        # Asistencia
        prob_att = models['attend'].predict_proba(test_df)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)
        
        # Rentabilidad y Segmentos
        res_prof = models['profit'].predict(test_df)[0]
        res_seg = models['segments'].predict(test_df.values)[0]
        
        # Revenue
        test_rev = np.array([[t_apuntados, avg_costo]])
        pred_rev = float(models['revenue'].predict(test_rev, verbose=0)[0][0])

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
        return jsonify({"error": f"Error en procesamiento: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
