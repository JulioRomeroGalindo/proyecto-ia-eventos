import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Habilitamos CORS para permitir peticiones desde tu App en AI Studio
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURACIÓN DE SEGURIDAD Y IA ---
# Leemos la API Key desde las variables de entorno de Render
API_KEY = os.environ.get('GEMINI_API_KEY')

if API_KEY:
    # IMPORTANTE: transport='rest' evita errores de gRPC/v1beta en entornos restringidos
    genai.configure(api_key=API_KEY, transport='rest')
else:
    print("⚠️ ADVERTENCIA: La variable GEMINI_API_KEY no está configurada en Render.")

# --- CARGA DE MODELOS DE MACHINE LEARNING ---
# Cargamos los modelos al inicio para que las respuestas sean instantáneas
try:
    model_attendance = joblib.load('model_attendance_v2.joblib')
    model_profitability = joblib.load('model_profitability_v2.joblib')
    model_segments = joblib.load('model_segments_v2.joblib')
    model_revenue = load_model('model_revenue_v2.h5')
    print("✅ Modelos v2 cargados exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar modelos: {str(e)}")

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard_kpis():
    try:
        # Lectura del CSV con los parámetros específicos solicitados
        df = pd.read_csv('REPORTE_MAESTRO_DEFINITIVO.csv', sep=';', encoding='utf-16')
        
        # Procesamiento y limpieza de columnas
        entran = pd.to_numeric(df['ENTRAN'], errors='coerce').fillna(0).sum()
        apuntados = pd.to_numeric(df['APUNTADOS'], errors='coerce').fillna(0).sum()
        ticket_prom = pd.to_numeric(df['VALOR_TICKET'], errors='coerce').fillna(0).mean()
        consumo_prom = pd.to_numeric(df['VALOR_CONSUMIBLE'], errors='coerce').fillna(0).mean()
        
        # Preparación de datos para los modelos
        # Asumimos que los modelos esperan el número de registrados como entrada principal
        input_data = np.array([[apuntados]])
        
        # Ejecución de Inferencia
        pred_asistencia = int(model_attendance.predict(input_data)[0])
        pred_rentabilidad = str(model_profitability.predict(input_data)[0])
        pred_segmento = str(model_segments.predict(input_data)[0])
        
        # Predicción de Revenue (Modelo Keras .h5)
        pred_revenue_raw = model_revenue.predict(input_data)
        pred_revenue = float(pred_revenue_raw[0][0])
        
        # Cálculo de Tasa de Conversión
        tasa_conv = (entran / apuntados * 100) if apuntados > 0 else 0
        
        return jsonify({
            "asistentes_reales": int(entran),
            "registrados": int(apuntados),
            "conversion": f"{tasa_conv:.2f}%",
            "pred_asistencia": pred_asistencia,
            "ticket_promedio": round(float(ticket_prom), 2),
            "rentabilidad": pred_rentabilidad,
            "perfil": pred_segmento,
            "revenue": f"${pred_revenue:,.2f}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Fallo en Dashboard: {str(e)}"}), 500

@app.route('/api/v1/chat', methods=['POST'])
def chat_consultant():
    try:
        payload = request.json
        pregunta = payload.get('pregunta')
        contexto = payload.get('contexto', {})
        # Permitimos que el frontend elija el modelo, pero por defecto usamos flash
        modelo_ia = payload.get('modelo', 'gemini-1.5-flash')

        if not pregunta:
            return jsonify({"error": "No se recibió ninguna pregunta"}), 400

        # Configuración del modelo Generativo
        model = genai.GenerativeModel(modelo_ia)
        
        prompt_ingenieria = f"""
        Eres un Consultor Estratégico de Eventos de alto nivel. 
        Analiza los siguientes datos reales del evento y responde la duda del usuario.
        
        DATOS ACTUALES DEL EVENTO:
        - Asistentes en Sala: {contexto.get('asistentes_reales', 'N/A')}
        - Registrados Totales: {contexto.get('registrados', 'N/A')}
        - Predicción de Asistencia Final (ML): {contexto.get('pred_asistencia', 'N/A')}
        - Nivel de Rentabilidad: {contexto.get('rentabilidad', 'N/A')}
        - Perfil de Audiencia Detectado: {contexto.get('perfil', 'N/A')}
        - Ingresos Proyectados (NN): {contexto.get('revenue', 'N/A')}
        
        PREGUNTA DEL USUARIO:
        {pregunta}
        
        INSTRUCCIONES:
        1. Responde de forma ejecutiva y basada en datos.
        2. Si la predicción es menor a los registrados, sugiere estrategias de asistencia.
        3. Si la rentabilidad es baja, propón ajustes en el ticket o consumibles.
        """
        
        response = model.generate_content(prompt_ingenieria)
        
        return jsonify({
            "respuesta": response.text,
            "modelo_usado": modelo_ia
        })

    except Exception as e:
        print(f"❌ Error en Chatbot: {str(e)}")
        # Fallback automático a gemini-pro si falla el modelo solicitado
        try:
            model_alt = genai.GenerativeModel('gemini-pro')
            response_alt = model_alt.generate_content(prompt_ingenieria)
            return jsonify({"respuesta": response_alt.text, "modelo_usado": "gemini-pro (fallback)"})
        except:
            return jsonify({"error": "El servicio de IA no está disponible temporalmente."}), 500

if __name__ == '__main__':
    # Render requiere que escuchemos en el puerto dinámico de la variable PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
