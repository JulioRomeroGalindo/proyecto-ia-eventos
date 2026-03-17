import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import google.generativeai as genai
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE GEMINI (FORZADO) ---
# Usamos transport='rest' para evitar que la librería busque rutas v1beta
genai.configure(api_key="AIzaSyAMkWJ5l6NZ1-g9znxNblGKDegQsWEAnGo", transport='rest')

# --- PARCHE PARA MODELO DE RED NEURONAL ---
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
        modelos["revenue"] = keras.models.load_model(
            ruta_h5, 
            custom_objects={"Dense": CustomDense},
            compile=False
        )
        print("✅ Modelos cargados correctamente")
        return modelos, None
    except Exception as e:
        print(f"❌ Error carga: {e}")
        return None, str(e)

MODELS, ERROR_MSG = cargar_recursos()

# --- RUTA DASHBOARD ---
@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    global MODELS, ERROR_MSG
    if MODELS is None:
        MODELS, ERROR_MSG = cargar_recursos()
        if MODELS is None: return jsonify({"error": str(ERROR_MSG)}), 500

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

# --- RUTA CHAT (EL CORAZÓN DEL PROBLEMA) ---
@app.route('/api/v1/chat', methods=['POST'])
def chat_interactivo():
    try:
        data_request = request.json
        pregunta = data_request.get("pregunta")
        contexto = data_request.get("contexto", {})

        # Usamos el alias '-latest' que es el más robusto contra errores 404
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"Actúa como consultor de eventos. Contexto técnico: {contexto}. Pregunta: {pregunta}"

        # La configuración 'rest' en genai.configure hará que esto use la API v1 directamente
        response = model.generate_content(prompt)
        
        if response and response.text:
            return jsonify({"respuesta": response.text})
        else:
            return jsonify({"respuesta": "El modelo no generó contenido. Intenta de nuevo."})

    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}")
        return jsonify({"respuesta": f"Fallo de conexión con Google AI: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
