import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- SISTEMA DE CARGA DINÁMICA ---
def cargar_recursos():
    # Buscamos en la carpeta actual y en la carpeta del script
    posibles_rutas = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    nombres = {
        "asistencia": "model_attendance_v2.joblib",
        "rentabilidad": "model_profitability_v2.joblib",
        "segmentos": "model_segments_v2.joblib",
        "revenue": "model_revenue_v2.h5"
    }
    
    modelos_cargados = {}
    
    try:
        for nick, archivo in nombres.items():
            encontrado = False
            for ruta in posibles_rutas:
                ruta_completa = os.path.join(ruta, archivo)
                if os.path.exists(ruta_completa):
                    if archivo.endswith('.h5'):
                        modelos_cargados[nick] = tf.keras.models.load_model(ruta_completa, compile=False)
                    else:
                        modelos_cargados[nick] = joblib.load(ruta_completa)
                    print(f"✅ Cargado: {archivo} desde {ruta}")
                    encontrado = True
                    break
            
            if not encontrado:
                print(f"❌ No se encontró: {archivo}")
                return None, f"Archivo faltante: {archivo}. Vistos: {os.listdir(posibles_rutas[0])}"
        
        return modelos_cargados, None
    except Exception as e:
        return None, str(e)

# Intentar carga global
MODELS, ERROR_MSG = cargar_recursos()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    global MODELS, ERROR_MSG
    # Re-intentar si falló al inicio
    if MODELS is None:
        MODELS, ERROR_MSG = cargar_recursos()
        if MODELS is None:
            return jsonify({"error": f"Modelos no listos: {ERROR_MSG}"}), 500

    try:
        # Carga del CSV con manejo de errores
        if not os.path.exists("REPORTE_MAESTRO_DEFINITIVO.csv"):
            return jsonify({"error": "CSV no encontrado en el servidor"}), 500
            
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float(df['COSTO_TOTAL'].mean())

        # Predicciones
        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        prob_att = MODELS["asistencia"].predict_proba(test_df)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)
        
        res_prof = MODELS["rentabilidad"].predict(test_df)[0]
        res_seg = MODELS["segmentos"].predict(test_df.values)[0]
        
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
        return jsonify({"error": f"Error en Dashboard: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
