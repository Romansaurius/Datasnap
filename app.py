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
    print("📥 [Render] Recibiendo petición POST en /procesar...")

    data = request.json
    print(f"📦 Datos recibidos: {data}")

    if not data or 'id' not in data:
        print("❌ No se envió el ID del archivo")
        return jsonify({"error": "No se envió el ID del archivo"}), 400

    id_archivo = data['id']

    try:
        print("🔗 Conectando a MySQL...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT ruta, nombre FROM archivos WHERE id = %s", (id_archivo,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            print(f"❌ Archivo con id {id_archivo} no encontrado en DB")
            return jsonify({"error": "Archivo no encontrado en la base de datos"}), 404

        ruta = result['ruta']
        print(f"📂 Archivo encontrado: {ruta}")

        if not os.path.exists(ruta):
            print(f"❌ Ruta no encontrada en servidor: {ruta}")
            return jsonify({"error": f"No se encontró el archivo en el servidor: {ruta}"}), 404

    except Exception as e:
        print(f"🔥 Error de conexión a MySQL: {e}")
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    extension = os.path.splitext(ruta)[1].lower()
    print(f"🛠 Procesando archivo con extensión {extension}")

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
            print("⚠ Formato no soportado")
            return jsonify({"error": "Formato no soportado"}), 400
    except Exception as e:
        print(f"🔥 Error procesando archivo: {e}")
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
    df.to_csv(salida, index=False, na_rep="NaN")
    print(f"✅ Archivo procesado y guardado en {salida}")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("UPDATE archivos SET estado = 'optimizado' WHERE id = %s", (id_archivo,))
        conn.commit()
        conn.close()
        print(f"🔄 Estado del archivo {id_archivo} actualizado a 'optimizado'")
    except Exception as e:
        print(f"🔥 Error actualizando estado en la DB: {e}")
        return jsonify({"error": f"No se pudo actualizar la base de datos: {e}"}), 500

    print("🚀 Proceso completado correctamente")
    return send_file(salida, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
