import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import requests  # Importante: para la comunicación directa
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- SEGURIDAD: LEEMOS LA LLAVE DESDE RENDER ---
api_key = os.environ.get("GEMINI_API_KEY")

# --- PARCHE PARA MODELOS ML (MANTENIDO) ---
@keras.saving.register_keras_serializable()
class CustomDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def cargar_recursos():
    base_path = os.path.dirname(os.path.abspath(__file__))
    modelos = {}
    try:
        modelos["asistencia"] = joblib.load(os.path.join(base_path, "model_attendance_v2.joblib"))
        modelos["rentabilidad"] = joblib.load(os.path.join(base_path, "model_profitability_v2.joblib"))
        modelos["segmentos"] = joblib.load(os.path.join(base_path, "model_segments_v2.joblib"))
        ruta_h5 = os.path.join(base_path, "model_revenue_v2.h5")
        modelos["revenue"] = keras.models.load_model(ruta_h5, custom_objects={"Dense": CustomDense}, compile=False)
        print("✅ Modelos de ML cargados con éxito")
        return modelos, None
    except Exception as e:
        print(f"❌ Error carga ML: {e}")
        return None, str(e)

MODELS, ERROR_MSG = cargar_recursos()

# --- RUTA 1: DASHBOARD (KPIs INTACTOS) ---
@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    if MODELS is None: return jsonify({"error": "Modelos no cargados"}), 500
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "REPORTE_MAESTRO_DEFINITIVO.csv")
        df = pd.read_csv(csv_path, sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float((df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']).mean())
        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        
        input_nn = np.array([[t_apuntados, avg_costo]], dtype="float32")
        pred_rev = float(np.array(MODELS["revenue"](input_nn))[0][0])
        
        return jsonify({
            "asistentes_reales": t_entran,
            "registrados": t_apuntados,
            "conversion": f"{(t_entran/t_apuntados*100):.2f}%" if t_apuntados > 0 else "0%",
            "pred_asistencia": int(t_apuntados * MODELS["asistencia"].predict_proba(test_df)[0][1]),
            "rentabilidad": "Optima" if MODELS["rentabilidad"].predict(test_df)[0] == 1 else "Baja",
            "perfil": "VIP" if MODELS["segmentos"].predict(test_df.values)[0] == 1 else "Estandar",
            "revenue": f"${pred_rev:,.2f}",
            "ticket_promedio": avg_costo 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RUTA 2: CHAT (SOLUCIÓN DE FUERZA BRUTA SIN LIBRERÍA GEMINI) ---
@app.route('/api/v1/chat', methods=['POST'])
def chat_interactivo():
    if not api_key:
        return jsonify({"respuesta": "Error: GEMINI_API_KEY no configurada en Render"}), 500
        
    try:
        data = request.json
        pregunta = data.get("pregunta")
        contexto = data.get("contexto", "")

        # LLAMADA DIRECTA A LA API V1 (SALTAMOS EL ERROR 404 V1BETA)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": f"Actúa como consultor experto en eventos. Contexto técnico: {contexto}. Pregunta: {pregunta}"}]
            }]
        }

        response = requests.post(url, json=payload, headers=headers)
        res_data = response.json()

        if response.status_code == 200:
            texto_ia = res_data['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"respuesta": texto_ia})
        else:
            error_msg = res_data.get('error', {}).get('message', 'Error desconocido de Google')
            return jsonify({"respuesta": f"Error de Google AI: {error_msg}"}), response.status_code

    except Exception as e:
        return jsonify({"respuesta": f"Fallo crítico de conexión: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
