from flask import Flask, request, jsonify, send_file
import os
import mysql.connector
from parsers.csv_parser import process_csv
from parsers.txt_parser import process_txt
from parsers.xlsx_parser import process_xlsx
from parsers.json_parser import process_json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
HISTORIAL_FOLDER = 'historial'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(HISTORIAL_FOLDER, exist_ok=True)

DB_CONFIG = {
    "host": "fedegoo.com.ar",
    "user": "9895",
    "password": "TIGRE.BC.9895",
    "database": "datasnap"
}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    file = request.files['file']
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    return jsonify({"success": True, "ruta": save_path})

@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "No se envió el ID del archivo"}), 400

    id_archivo = data['id']

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT ruta, nombre FROM archivos WHERE id = %s", (id_archivo,))
        result = cursor.fetchone()
        conn.close()
        if not result:
            return jsonify({"error": "Archivo no encontrado en la base de datos"}), 404
        ruta = result['ruta']
        if not os.path.exists(ruta):
            return jsonify({"error": f"No se encontró el archivo en Render: {ruta}"}), 404
    except Exception as e:
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    extension = os.path.splitext(ruta)[1].lower()
    try:
        if extension == ".csv":
            df = process_csv(ruta, HISTORIAL_FOLDER)
        elif extension == ".txt":
            df = process_txt(ruta, HISTORIAL_FOLDER)
        elif extension == ".xlsx":
            df = process_xlsx(ruta, HISTORIAL_FOLDER)
        elif extension == ".json":
            df = process_json(ruta, HISTORIAL_FOLDER)
        else:
            return jsonify({"error": "Formato no soportado"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
    df.to_csv(salida, index=False, na_rep="NaN")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("UPDATE archivos SET estado = 'optimizado' WHERE id = %s", (id_archivo,))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"No se pudo actualizar la base de datos: {e}"}), 500

    return send_file(salida, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
