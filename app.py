from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json, mysql.connector, requests, time, logging
from parsers.csv_parser import process_csv
from parsers.txt_parser import process_txt
from parsers.xlsx_parser import process_xlsx
from parsers.json_parser import process_json
from parsers.sql_parser import process_sql
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
HISTORIAL_FOLDER = 'historial'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(HISTORIAL_FOLDER, exist_ok=True)

# Config DB desde variables de entorno
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "port": int(os.environ.get("DB_PORT", 3306))
}

# Config Google Drive
try:
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable not found")
    
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=creds)
    
    DRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")
    if not DRIVE_FOLDER_ID:
        raise ValueError("GDRIVE_FOLDER_ID environment variable not found")
        
    logging.info("Google Drive configurado correctamente")
except Exception as e:
    logging.error(f"Error configurando Google Drive: {e}")
    drive_service = None
    DRIVE_FOLDER_ID = None

def execute_with_retry(request):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return request.execute()
        except Exception as e:
            if 'quotaExceeded' in str(e) or 'quota_exceeded' in str(e).lower():
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise
    raise Exception("Max retries exceeded for Google Drive API quota")

@app.route('/upload_original', methods=['POST'])
def upload_original():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No se envió archivo"}), 400

    file = request.files['file']
    local_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(local_path)

    try:
        file_metadata = {"name": file.filename, "parents": [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(local_path, mimetype="application/octet-stream")
        uploaded = execute_with_retry(drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink"
        ))

        drive_id = uploaded["id"]
        drive_link = uploaded["webViewLink"]

        execute_with_retry(drive_service.permissions().create(
            fileId=drive_id,
            body={"type": "anyone", "role": "reader"}
        ))

        return jsonify({"success": True, "drive_id": drive_id, "drive_link": drive_link})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
    if not drive_service:
        return jsonify({"error": "Google Drive no está configurado correctamente"}), 500
        
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "No se envió el ID del archivo"}), 400

    id_archivo = data['id']
    logging.info("Procesando archivo ID: %s", id_archivo)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT ruta, nombre, drive_id_original FROM archivos WHERE id = %s AND estado != 'borrado'", (id_archivo,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"error": "Archivo no encontrado en la base de datos"}), 404

        ruta = result['ruta']
        temp_file = False

        if result.get('drive_id_original'):
            try:
                file_info = drive_service.files().get(fileId=result['drive_id_original'], fields="id,name,parents").execute()
                logging.info(f"Archivo encontrado en Drive: {file_info}")
                
                drive_request = drive_service.files().get_media(fileId=result['drive_id_original'])
                temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{id_archivo}_{result['nombre']}")
                with open(temp_path, 'wb') as f:
                    f.write(execute_with_retry(drive_request))
                ruta = temp_path
                temp_file = True
                logging.info(f"Archivo descargado exitosamente a: {temp_path}")
            except Exception as e:
                logging.error(f"Error accediendo a Drive: {e}")
                if result['ruta'] and os.path.exists(result['ruta']):
                    ruta = result['ruta']
                else:
                    return jsonify({"error": f"Archivo no accesible: {e}"}), 404
        else:
            if not result['ruta'] or not os.path.exists(result['ruta']):
                return jsonify({"error": f"No se encontró el archivo local: {result['ruta']}"}), 404
            ruta = result['ruta']
    except Exception as e:
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    extension = os.path.splitext(ruta)[1].lower()
    logging.info("Procesando archivo con extensión: %s, ruta: %s", extension, ruta)
    
    try:
        if extension == ".csv":
            df = process_csv(ruta, HISTORIAL_FOLDER)
            salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
            df.to_csv(salida, index=False, na_rep="NaN")
        elif extension == ".txt":
            df = process_txt(ruta, HISTORIAL_FOLDER)
            salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
            df.to_csv(salida, index=False, na_rep="NaN")
        elif extension == ".xlsx":
            df = process_xlsx(ruta, HISTORIAL_FOLDER)
            salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
            df.to_csv(salida, index=False, na_rep="NaN")
        elif extension == ".json":
            df = process_json(ruta, HISTORIAL_FOLDER)
            salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
            df.to_csv(salida, index=False, na_rep="NaN")
        elif extension == ".sql":
            optimized_sql = process_sql(ruta, HISTORIAL_FOLDER)
            salida = os.path.join(PROCESSED_FOLDER, f"optimizado_{os.path.basename(ruta)}")
            with open(salida, 'w', encoding='utf-8') as f:
                f.write(optimized_sql)
        else:
            return jsonify({"error": "Formato no soportado"}), 400
        
        logging.info("Procesamiento completado, archivo guardado en: %s", salida)
    except Exception as e:
        logging.error("Error al procesar: %s", e)
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    # Limpiar archivo temporal
    if temp_file:
        try:
            os.remove(ruta)
            logging.info(f"Archivo temporal eliminado: {ruta}")
        except Exception as e:
            logging.warning(f"No se pudo eliminar archivo temporal: {e}")

    # Subir archivo procesado a Drive
    try:
        file_metadata = {"name": f"procesado_{os.path.basename(ruta)}", "parents": [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(salida, mimetype="application/octet-stream")
        uploaded = execute_with_retry(drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink"
        ))

        drive_id = uploaded["id"]
        drive_link = uploaded["webViewLink"]

        execute_with_retry(drive_service.permissions().create(
            fileId=drive_id,
            body={"type": "anyone", "role": "reader"}
        ))

        # Actualizar DB
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE archivos SET estado = 'procesado', drive_id_procesado = %s, drive_link_procesado = %s WHERE id = %s",
            (drive_id, drive_link, id_archivo)
        )
        conn.commit()
        conn.close()

        # Leer contenido para respuesta
        if extension == ".sql":
            with open(salida, 'r', encoding='utf-8') as f:
                contenido = f.read()
        else:
            contenido = "Archivo procesado exitosamente"

        return jsonify({
            "success": True,
            "drive_id": drive_id,
            "drive_link": drive_link,
            "archivo_local": salida,
            "contenido": contenido
        })
    except Exception as e:
        logging.error("Error subiendo a Drive: %s", e)
        return jsonify({"error": f"Error subiendo archivo procesado: {e}"}), 500

@app.route('/descargar/<int:id_archivo>', methods=['GET'])
def descargar(id_archivo):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT drive_id_procesado, nombre FROM archivos WHERE id = %s AND estado = 'procesado'", (id_archivo,))
        result = cursor.fetchone()
        conn.close()

        if not result or not result['drive_id_procesado']:
            return jsonify({"error": "Archivo procesado no encontrado"}), 404

        drive_request = drive_service.files().get_media(fileId=result['drive_id_procesado'])
        temp_path = os.path.join(PROCESSED_FOLDER, f"descarga_{result['nombre']}")
        
        with open(temp_path, 'wb') as f:
            f.write(execute_with_retry(drive_request))

        return send_file(temp_path, as_attachment=True, download_name=f"procesado_{result['nombre']}")
    except Exception as e:
        return jsonify({"error": f"Error descargando archivo: {e}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "drive_configured": drive_service is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)