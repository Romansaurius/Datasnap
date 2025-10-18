from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
from datetime import datetime
import traceback

# Importar la IA Global Universal
from optimizers.universal_global_ai import UniversalGlobalAI

app = Flask(__name__)

# Instancia de la IA Global
global_ai = UniversalGlobalAI()

@app.route('/procesar', methods=['POST'])
def procesar():
    """Procesa archivos con IA Global Universal"""
    try:
        data = request.get_json()
        file_id = data.get('id')
        google_token = data.get('google_refresh_token')
        
        if not file_id:
            return jsonify({'success': False, 'error': 'ID de archivo requerido'}), 400
        
        print(f"🤖 Procesando archivo ID: {file_id} con IA Global Universal")
        
        # Simular datos de prueba (reemplaza con tu lógica de BD)
        test_data = pd.DataFrame({
            'nombre': ['Juan Perez', 'maria garcia', 'nan', 'Pedro López', ''],
            'email': ['juan@gmai.com', 'maria@hotmial.com', 'pedro@yahoo.co', 'ana@gmail.com', 'carlos'],
            'edad': [25, 'NaN', 30, 22, '35'],
            'precio': ['NaN', 200, 'NaN', 150, 'NaN'],
            'activo': ['si', '1', 'nan', 'true', 'false']
        })
        
        print(f"📊 Datos originales: {len(test_data)} filas")
        
        # 🌍 APLICAR IA GLOBAL UNIVERSAL
        optimized_data = global_ai.process_any_data(test_data)
        
        print(f"✅ Datos optimizados: {len(optimized_data)} filas")
        
        # Generar estadísticas
        stats = {
            'filas_originales': len(test_data),
            'filas_optimizadas': len(optimized_data),
            'duplicados_eliminados': len(test_data) - len(optimized_data),
            'ia_global_aplicada': True,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"📈 Estadísticas: {stats}")
        
        return jsonify({
            'success': True,
            'message': 'Archivo optimizado con IA Global Universal',
            'estadisticas': stats
        })
        
    except Exception as e:
        error_msg = f"Error procesando archivo: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/upload_original', methods=['POST'])
def upload_original():
    """Sube archivo original a Google Drive"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        google_token = request.form.get('google_refresh_token')
        
        if not google_token:
            return jsonify({'success': False, 'error': 'Google token required'}), 400
        
        # Simular subida exitosa (reemplaza con tu lógica de Google Drive)
        drive_id = f"drive_id_{int(datetime.now().timestamp())}"
        drive_link = f"https://drive.google.com/file/d/{drive_id}/view"
        
        print(f"📤 Archivo subido: {file.filename} -> {drive_id}")
        
        return jsonify({
            'success': True,
            'drive_id': drive_id,
            'drive_link': drive_link
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check con IA Global"""
    return jsonify({
        'status': 'ok',
        'ia_global_disponible': True,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🌍 Iniciando Render con IA Global Universal...")
    app.run(host='0.0.0.0', port=port)
