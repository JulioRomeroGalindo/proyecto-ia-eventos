import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CARGA DE RECURSOS CON LIMPIEZA ---
def cargar_y_limpiar():
    try:
        # Intentamos leer con separador punto y coma (común en Excel español)
        df = pd.read_csv("REPORTE_MAESTRO_DEFINITIVO.csv", sep=';', encoding='utf-16')
        
        # LIMPIEZA CRÍTICA: Convertir columnas a números, eliminando espacios o basura
        for col in ['ENTRAN', 'APUNTADOS', 'VALOR_TICKET']:
            if col in df.columns:
                # Quitamos puntos de miles, cambiamos comas decimales por puntos, y convertimos
                df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return pd.DataFrame()

# Cargar modelo
try:
    m_attend = joblib.load('model_attendance.joblib')
except:
    m_attend = None

@app.route('/api/v1/kpi/dashboard', methods=['GET'])
def get_dashboard_kpis():
    try:
        df = cargar_y_limpiar()
        if df.empty:
            return jsonify({"error": "CSV vacío o no encontrado"}), 500

        total_entran = int(df['ENTRAN'].sum())
        total_apuntados = int(df['APUNTADOS'].sum())
        
        # --- LÓGICA DE PREDICCIÓN CORREGIDA ---
        if m_attend is not None:
            # Usamos el total de registrados como entrada
            input_data = np.array([[total_apuntados]])
            pred_val = m_attend.predict(input_data)[0]
            
            # Si el modelo devuelve algo menor a los que ya entraron, 
            # lo ajustamos para que tenga sentido lógico.
            asistencia_final = max(int(pred_val), total_entran)
            
            # Si a pesar de todo el modelo devuelve 0 pero hay gente apuntada,
            # aplicamos una regla de negocio simple para no mostrar 0.
            if asistencia_final == 0 and total_apuntados > 0:
                asistencia_final = int(total_apuntados * 0.85) # Estimación lógica
        else:
            asistencia_final = total_entran

        tasa = (total_entran / total_apuntados * 100) if total_apuntados > 0 else 0

        return jsonify({
            "total_invitados": total_entran,
            "total_registrados": total_apuntados,
            "tasa_conversion": f"{tasa:.2f}%",
            "prediccion_asistencia": asistencia_final,
            "ticket_promedio": int(df['VALOR_TICKET'].mean() if 'VALOR_TICKET' in df else 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
