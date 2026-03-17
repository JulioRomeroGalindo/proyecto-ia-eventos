import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CARGA DE MODELOS V2 ---
try:
    # Usamos los nombres exactos de tus nuevos archivos
    m_attend = joblib.load('model_attendance_v2.joblib')
    m_profit = joblib.load('model_profitability_v2.joblib')
    m_segments = joblib.load('model_segments_v2.joblib')
    m_revenue = tf.keras.models.load_model('model_revenue_v2.h5', compile=False)
    print("✅ Los 4 modelos V2 cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")

def get_clean_data():
    try:
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET', 'VALOR_CONSUMIBLE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        # Regra de negocio unificada
        df['COSTO_TOTAL'] = df['VALOR_TICKET'] + df['VALOR_CONSUMIBLE']
        return df
    except:
        return pd.DataFrame()

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard():
    try:
        df = get_clean_data()
        t_apuntados = int(df['APUNTADOS'].sum())
        t_entran = int(df['ENTRAN'].sum())
        avg_costo = df['COSTO_TOTAL'].mean()

        # --- PREDICCIONES CON MODELOS V2 ---
        
        # 1. Asistencia (Probabilidad escalada)
        # El modelo v2 devuelve probabilidad, multiplicamos por registrados
        test_att = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        prob_att = m_attend.predict_proba(test_att)[0][1]
        pred_asistencia = int(t_apuntados * prob_att)

        # 2. Rentabilidad
        test_prof = pd.DataFrame([[avg_costo]], columns=['COSTO_TOTAL'])
        res_prof = m_profit.predict(test_prof)[0]
        label_rentabilidad = "Óptima" if res_prof == 1 else "Baja"

        # 3. Segmentación
        # Quitamos nombres de columna con .values para evitar warnings
        res_seg = m_segments.predict(np.array([[avg_costo]]))[0]
        nombres_segmentos = {0: "Estándar", 1: "VIP / Premium", 2: "Base / Masivo"}
        label_segmento = nombres_segmentos.get(res_seg, "Diversificado")

        # 4. Revenue (Red Neuronal)
        test_rev = np.array([[t_apuntados, avg_costo]])
        pred_rev = float(m_revenue.predict(test_rev, verbose=0)[0][0])

        return jsonify({
            "total_invitados": t_entran,
            "total_registrados": t_apuntados,
            "tasa_conversion": f"{(t_entran/t_apuntados*100):.2f}%" if t_apuntados > 0 else "0%",
            "prediccion_asistencia": pred_asistencia,
            "ticket_promedio": int(avg_costo),
            "rentabilidad": label_rentabilidad,
            "perfil_evento": label_segmento,
            "revenue_estimado": f"${pred_rev:,.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
