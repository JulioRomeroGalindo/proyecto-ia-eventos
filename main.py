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

# --- CONFIGURACIÓN DE GEMINI ---
try:
    # Tu API Key detectada
    genai.configure(api_key="AIzaSyAMkWJ5l6NZ1-g9znxNblGKDegQsWEAnGo")
    # Modelo estable
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
    print("✅ Configuración de Gemini preparada")
except Exception as e:
    print(f"❌ Error configurando Gemini: {e}")

# --- PARCHE DE EMERGENCIA PARA DENSE LAYER ---
@keras.saving.register_keras_serializable()
class CustomDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

def cargar_recursos():
    base_path = os.path.dirname(os.path.abspath(__file__))
    modelos = {}
    try:
        # Carga de modelos Scikit-Learn
        modelos["asistencia"] = joblib.load(os.path.join(base_path, "model_attendance_v2.joblib"))
        modelos["rentabilidad"] = joblib.load(os.path.join(base_path, "model_profitability_v2.joblib"))
        modelos["segmentos"] = joblib.load(os.path.join(base_path, "model_segments_v2.joblib"))
        
        # Carga de Red Neuronal con parche aplicado
        ruta_h5 = os.path.join(base_path, "model_revenue_v2.h5")
        modelos["revenue"] = keras.models.load_model(
            ruta_h5, 
            custom_objects={"Dense": CustomDense},
            compile=False
        )
        print("✅ SISTEMA OPERATIVO: Modelos cargados con éxito.")
        return modelos, None
    except Exception as e:
        print(f"❌ Error carga: {e}")
        return None, str(e)

MODELS, ERROR_MSG = cargar_recursos()

# --- RUTA 1: DASHBOARD (AUDITORÍA) ---
@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    global MODELS, ERROR_MSG
    if MODELS is None:
        MODELS, ERROR_MSG = cargar_recursos()
        if MODELS is None: return jsonify({"error": f"Error crítico: {ERROR_MSG}"}), 500

    try:
        csv_path = os.path.join(os.path.dirname(__file__), "REPORTE_MAESTRO_DEFINITIVO.csv")
        df = pd.read_csv(csv_path, sep=';', encoding='utf-16')
        
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = float((df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']).mean())

        test_df = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        p_asistencia = int(t_apuntados * MODELS["asistencia"].predict_proba(test_df)[0][1])
        res_prof = "Optima" if MODELS["rentabilidad"].predict(test_df)[0] == 1 else "Baja"
        res_seg = "VIP" if MODELS["segmentos"].predict(test_df.values)[0] == 1 else "Estandar"
        
        input_nn = np.array([[t_apuntados, avg_costo]], dtype="float32")
        pred_rev = float(np.array(MODELS["revenue"](input_nn))[0][0])

        return jsonify({
            "asistentes_reales": t_entran,
            "registrados": t_apuntados,
            "conversion": f"{(t_entran/t_apuntados*100):.2f}%" if t_apuntados > 0 else "0%",
            "pred_asistencia": p_asistencia,
            "rentabilidad": res_prof,
            "perfil": res_seg,
            "revenue": f"${pred_rev:,.2f}",
            "ticket_promedio": avg_costo 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RUTA 2: CHATBOT INTERACTIVO (ÚNICA INSTANCIA) ---
@app.route('/api/v1/chat', methods=['POST'])
def chat_interactivo():
    try:
        data_request = request.json
        pregunta_usuario = data_request.get("pregunta")
        contexto_modelos = data_request.get("contexto", {})

        prompt = f"""
        Actúa como un Consultor Estratégico de Eventos con IA.
        DATOS DE MODELOS DE MACHINE LEARNING:
        - Registrados actual: {contexto_modelos.get('registrados')}
        - Asistencia Predicha: {contexto_modelos.get('pred_asistencia')}
        - Perfil: {contexto_modelos.get('perfil')}
        - Ingresos: {contexto_modelos.get('revenue')}
        - Ticket Promedio: {contexto_modelos.get('ticket_promedio')}

        PREGUNTA DEL USUARIO: "{pregunta_usuario}"

        INSTRUCCIÓN: Responde usando los datos técnicos. Si proponen cambios (ej. 5000 personas), 
        proyecta resultados basados en la tasa de conversión y ticket promedio actual.
        """

        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            return jsonify({"respuesta": response.text})
        else:
            return jsonify({"respuesta": "La IA procesó la duda pero no generó texto. Intenta de nuevo."})
    
    except Exception as e:
        return jsonify({"respuesta": f"Error en el cerebro de IA: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
