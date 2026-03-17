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

# --- CONFIGURACIÓN DE GEMINI (CON FIX DE RUTA) ---
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    # transport='rest' obliga a usar la ruta v1 estable y evita la v1beta que falla
    genai.configure(api_key=api_key, transport='rest')
    print("✅ API Gemini configurada correctamente")
else:
    print("❌ ERROR: No se encontró GEMINI_API_KEY en Environment")

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
        print("✅ Modelos de ML cargados")
        return modelos, None
    except Exception as e:
        print(f"❌ Error carga ML: {e}")
        return None, str(e)

MODELS, ERROR_MSG = cargar_recursos()

# --- RUTA 1: DASHBOARD (KPIs MANTENIDOS) ---
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
        
        return jsonify({
            "asistentes_reales": t_entran,
            "registrados": t_apuntados,
            "conversion": f"{(t_entran/t_apuntados*100):.2f}%" if t_apuntados > 0 else "0%",
            "pred_asistencia": int(t_apuntados * MODELS["asistencia"].predict_proba(test_df)[0][1]),
            "rentabilidad": "Optima" if MODELS["rentabilidad"].predict(test_df)[0] == 1 else "Baja",
            "perfil": "VIP" if MODELS["segmentos"].predict(test_df.values)[0] == 1 else "Estandar",
            "revenue": f"${float(np.array(MODELS['revenue'](np.array([[t_apuntados, avg_costo]], dtype='float32')))[0][0]):,.2f}",
            "ticket_promedio": avg_costo 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RUTA 2: CHAT (CORREGIDO) ---
@app.route('/api/v1/chat', methods=['POST'])
def chat_interactivo():
    try:
        data = request.json
        # 'gemini-1.5-flash' es el nombre más compatible con la ruta v1
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"Actúa como consultor experto. Contexto: {data.get('contexto')}. Pregunta: {data.get('pregunta')}"
        response = model.generate_content(prompt)
        
        return jsonify({"respuesta": response.text})
    except Exception as e:
        # Si esto falla, nos dirá exactamente por qué (región, llave o versión)
        return jsonify({"respuesta": f"Error de conexión: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
