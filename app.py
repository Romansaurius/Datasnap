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

@app.route('/procesar', methods=['POST'])
def procesar():
    # 1. Recibir ID o nombre de archivo desde la petición POST
    data = request.json
    if not data or 'archivo' not in data:
        return jsonify({"error": "No se envió el nombre o ID de archivo"}), 400

    nombre_archivo = data['archivo']

    # 2. Buscar archivo en la base de datos y traer su contenido
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT contenido, extension FROM archivos WHERE nombre = %s", (nombre_archivo,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({"error": "Archivo no encontrado en la base de datos"}), 404

        contenido, extension = result
        ruta_entrada = os.path.join(UPLOAD_FOLDER, nombre_archivo)
        with open(ruta_entrada, 'wb') as f:
            f.write(contenido)

    except Exception as e:
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    # 3. Procesar el archivo según extensión
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
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    # 4. Guardar archivo procesado en carpeta y actualizar estado en la BD
    salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{nombre_archivo}.csv")
    df.to_csv(salida, index=False, na_rep="NaN")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        with open(salida, 'rb') as f:
            contenido_mejorado = f.read()
        cursor.execute("UPDATE archivos SET contenido = %s, estado = 'optimizado' WHERE nombre = %s", (contenido_mejorado, nombre_archivo))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"No se pudo actualizar la base de datos: {e}"}), 500

    return send_file(salida, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
