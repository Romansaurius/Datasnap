from flask import Flask, request, jsonify
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

# Configuración de conexión a la base de datos de Hostinger
DB_CONFIG = {
    "host": "fedegoo.com.ar",  # ejemplo: "mysql.hostinger.com"
    "user": "9895",
    "password": "TIGRE.BC.9895",
    "database": "datasnap"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

@app.route('/procesar', methods=['GET'])
def procesar():
    archivo_id = request.args.get("id")
    if not archivo_id:
        return jsonify({"error": "Falta el parámetro 'id'"}), 400

    try:
        conexion = get_db_connection()
        cursor = conexion.cursor(dictionary=True)

        # Buscar archivo en la DB
        cursor.execute("SELECT * FROM archivos WHERE id = %s", (archivo_id,))
        archivo = cursor.fetchone()

        if not archivo:
            return jsonify({"error": "Archivo no encontrado"}), 404

        ruta_entrada = archivo['ruta']
        nombre = archivo['nombre']
        user_id = archivo['user_id']

        if not os.path.exists(ruta_entrada):
            return jsonify({"error": f"El archivo no existe en {ruta_entrada}"}), 404

        extension = os.path.splitext(nombre)[1].lower()

        try:
            if extension == ".csv":
                df = process_csv(ruta_entrada, HISTORIAL_FOLDER)
            elif extension == ".txt":
                df = process_txt(ruta_entrada, HISTORIAL_FOLDER)
            elif extension == ".xlsx":
                df = process_xlsx(ruta_entrada, HISTORIAL_FOLDER)
            elif extension == ".json":
                df = process_json(ruta_entrada, HISTORIAL_FOLDER)
            else:
                return jsonify({"error": "Formato no soportado"}), 400
        except Exception as e:
            return jsonify({"error": f"Error al procesar archivo: {e}"}), 500

        # Guardar archivo procesado
        nombre_salida = f"mejorado_{nombre.replace(extension, '.csv')}"
        salida = os.path.join(PROCESSED_FOLDER, nombre_salida)
        df.to_csv(salida, index=False, na_rep="NaN")

        # Registrar archivo optimizado en la base de datos
        cursor.execute("""
            INSERT INTO archivos (user_id, nombre, ruta, estado)
            VALUES (%s, %s, %s, 'optimizado')
        """, (user_id, nombre_salida, salida))
        conexion.commit()

        return jsonify({"status": "ok", "mensaje": "Archivo optimizado", "archivo": nombre_salida})

    except mysql.connector.Error as db_err:
        return jsonify({"error": f"Error de base de datos: {db_err}"}), 500
    finally:
        if conexion.is_connected():
            cursor.close()
            conexion.close()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
