from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json, mysql.connector, requests, time, logging
from optimizers.master_universal_processor import MasterUniversalProcessor
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API funcionando", "message": "DataSnap API activa"})

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
    creds_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    logging.info("Credenciales JSON cargadas correctamente")
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=creds)
    DRIVE_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
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
                wait_time = 2 ** attempt  # exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise
    raise Exception("Max retries exceeded for Google Drive API quota")

@app.route('/upload_original', methods=['POST'])
def upload_original():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No se envió archivo"}), 400

    google_refresh_token = request.form.get('google_refresh_token')
    if not google_refresh_token:
        return jsonify({"success": False, "error": "Token de Google requerido"}), 400

    file = request.files['file']
    local_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(local_path)

    try:
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
        user_creds = Credentials(
            None,
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=client_id,
            client_secret=client_secret
        )
        if user_creds.expired:
            user_creds.refresh(Request())
        user_drive_service = build("drive", "v3", credentials=user_creds)

        file_metadata = {"name": file.filename}
        media = MediaFileUpload(local_path, mimetype="application/octet-stream")
        uploaded = execute_with_retry(user_drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink"
        ))

        drive_id = uploaded["id"]
        drive_link = uploaded["webViewLink"]

        return jsonify({"success": True, "drive_id": drive_id, "drive_link": drive_link})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/test_db', methods=['GET'])
def test_db():
    try:
        logging.info(f"Testing DB connection: host={DB_CONFIG['host']}, user={DB_CONFIG['user']}, db={DB_CONFIG['database']}, port={DB_CONFIG['port']}")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DESCRIBE archivos")
        columns = cursor.fetchall()
        conn.close()
        return jsonify({"status": "ok", "message": "BD conectada", "columns": [col[0] for col in columns]})
    except Exception as e:
        logging.error(f"Error de conexión BD: {e}")
        return jsonify({"status": "error", "message": f"Error BD: {str(e)}"})

@app.route('/test_drive', methods=['GET'])
def test_drive():
    if not drive_service or not DRIVE_FOLDER_ID:
        return jsonify({"status": "error", "message": "Google Drive no configurado"})
    
    try:
        # Test simple: listar archivos en la carpeta
        results = drive_service.files().list(q=f"'{DRIVE_FOLDER_ID}' in parents", pageSize=1).execute()
        return jsonify({"status": "ok", "message": "Google Drive configurado correctamente"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error en Google Drive: {str(e)}"})

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
    google_refresh_token = data.get('google_refresh_token')
    logging.info("Procesando archivo ID: %s", id_archivo)

    # Setup Drive service
    if google_refresh_token:
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
        user_creds = Credentials(
            None,
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=client_id,
            client_secret=client_secret
        )
        if user_creds.expired:
            user_creds.refresh(Request())
        drive_service_to_use = build("drive", "v3", credentials=user_creds)
        drive_folder = ['root']
    else:
        drive_service_to_use = drive_service
        drive_folder = [DRIVE_FOLDER_ID]

    try:
        logging.info(f"Intentando conectar a BD: host={DB_CONFIG['host']}, user={DB_CONFIG['user']}, db={DB_CONFIG['database']}, port={DB_CONFIG['port']}")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT ruta, nombre, drive_id_original FROM archivos WHERE id = %s", (id_archivo,))
        result = cursor.fetchone()
        conn.close()
        logging.info("Resultado DB: %s", result)
        if not result:
            return jsonify({"error": "Archivo no encontrado en la base de datos"}), 404

        # SIEMPRE descargar desde Google Drive (nunca usar ruta local de Hostinger)
        if not result.get('drive_id_original'):
            return jsonify({"error": "Archivo no disponible en Google Drive. Debe subirse primero."}), 404
        
        # Descargar desde Google Drive usando API
        try:
            drive_request = drive_service_to_use.files().get_media(fileId=result['drive_id_original'])
            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{id_archivo}_{result['nombre']}")
            with open(temp_path, 'wb') as f:
                f.write(execute_with_retry(drive_request))
            ruta = temp_path
            temp_file = True
            logging.info("Archivo descargado desde Google Drive: %s", temp_path)
        except Exception as e:
            logging.error("Error descargando desde Google Drive: %s", e)
            return jsonify({"error": f"No se pudo descargar el archivo desde Google Drive: {e}"}), 500
    except Exception as e:
        logging.error(f"Error de conexión BD: {e}")
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    extension = os.path.splitext(ruta)[1].lower()
    logging.info("Procesando archivo con IA GLOBAL UNIVERSAL: %s, ruta: %s", extension, ruta)
    
    try:
        # USAR PROCESADOR MAESTRO UNIVERSAL
        master_processor = MasterUniversalProcessor()
        processing_result = master_processor.process_any_file(ruta, HISTORIAL_FOLDER)
        
        if not processing_result['success']:
            error_msg = '; '.join(processing_result['errors']) if processing_result['errors'] else 'Error desconocido'
            return jsonify({"error": f"Error procesando archivo: {error_msg}"}), 500
        
        processed_data = processing_result['processed_data']
        detected_type = processing_result['file_info'].get('detected_type', 'unknown')
        
        # Determinar extensión de salida
        if detected_type == 'sql':
            output_extension = ".sql"
            df = None
            optimized_sql = processed_data
        else:
            output_extension = ".csv"
            df = processed_data if isinstance(processed_data, pd.DataFrame) else pd.DataFrame({'data': [str(processed_data)]})
            optimized_sql = None
        
        logging.info("Procesamiento completado con IA Global Universal. Tipo detectado: %s", detected_type)
        if df is not None:
            logging.info("DataFrame resultante: %d filas, %d columnas", len(df), len(df.columns))
        else:
            logging.info("Contenido SQL optimizado generado")
    except Exception as e:
        logging.error("Error al procesar: %s", e)
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    # Guardar archivo procesado
    base_name = os.path.splitext(os.path.basename(ruta))[0]
    salida = os.path.join(PROCESSED_FOLDER, f"optimizado_{base_name}{output_extension}")
    
    if df is not None:
        # Para archivos tabulares (CSV, XLSX, etc.)
        df.to_csv(salida, index=False, na_rep="")
    else:
        # Para archivos SQL u otros formatos de texto
        with open(salida, 'w', encoding='utf-8') as f:
            f.write(str(optimized_sql))
    
    logging.info("Archivo procesado guardado en: %s", salida)

    # Limpiar archivo temporal si se descargó
    if temp_file:
        os.remove(ruta)
        logging.info("Archivo temporal eliminado")

    # Subir a Google Drive
    try:
        logging.info("Subiendo a Google Drive: %s", salida)
        file_metadata = {
            "name": os.path.basename(salida),
            "parents": drive_folder
        }
        
        # Determinar MIME type según extensión
        if output_extension == ".sql":
            mimetype = "text/plain"
        else:
            mimetype = "text/csv"
            
        media = MediaFileUpload(salida, mimetype=mimetype)
        uploaded = execute_with_retry(drive_service_to_use.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink, webContentLink"
        ))

        drive_id = uploaded.get("id")
        drive_link = uploaded.get("webViewLink")
        logging.info("Subido a Drive, ID: %s", drive_id)

        # Hacer accesible por link público
        execute_with_retry(drive_service_to_use.permissions().create(
            fileId=drive_id,
            body={"type": "anyone", "role": "reader"}
        ))

    except Exception as e:
        logging.error("Error subiendo a Drive: %s", e)
        return jsonify({"error": f"No se pudo subir a Google Drive: {e}"}), 500

    # Actualizar base de datos
    try:
        logging.info("Actualizando DB para ID: %s", id_archivo)
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE archivos
            SET estado = 'optimizado',
                drive_id_optimizado = %s,
                drive_link_optimizado = %s,
                fecha_optimizacion = NOW()
            WHERE id = %s
        """, (drive_id, drive_link, id_archivo))
        conn.commit()
        conn.close()
        logging.info("DB actualizada exitosamente")
    except Exception as e:
        logging.error("Error actualizando DB: %s", e)
        return jsonify({"error": f"No se pudo actualizar la base de datos: {e}"}), 500

    # Leer el contenido del archivo para enviarlo de vuelta
    with open(salida, 'r', encoding='utf-8') as f:
        archivo_optimizado = f.read()
    
    return jsonify({
        "success": True,
        "archivo_id": id_archivo,
        "ruta_local": salida,
        "drive_id": drive_id,
        "drive_link": drive_link,
        "archivo_optimizado": archivo_optimizado,
        "nombre_archivo": os.path.basename(salida)
    })

@app.route('/procesar_drive', methods=['POST'])
def procesar_drive():
    data = request.json
    if not data or 'drive_file_id' not in data:
        return jsonify({"error": "No se envió el ID del archivo de Google Drive"}), 400

    drive_file_id = data['drive_file_id']

    try:
        # Download from Drive
        request = drive_service.files().get_media(fileId=drive_file_id)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_drive_{drive_file_id}")
        with open(temp_path, 'wb') as f:
            f.write(execute_with_retry(request))

        # Get file name
        file_info = execute_with_retry(drive_service.files().get(fileId=drive_file_id, fields="name"))
        nombre = file_info['name']
        ruta = temp_path
        temp_file = True
    except Exception as e:
        return jsonify({"error": f"Error al descargar desde Google Drive: {e}"}), 500

    extension = os.path.splitext(nombre)[1].lower()
    try:
        # USAR PROCESADOR MAESTRO UNIVERSAL
        master_processor = MasterUniversalProcessor()
        processing_result = master_processor.process_any_file(ruta, HISTORIAL_FOLDER)
        
        if not processing_result['success']:
            if temp_file:
                os.remove(ruta)
            error_msg = '; '.join(processing_result['errors']) if processing_result['errors'] else 'Error desconocido'
            return jsonify({"error": f"Error procesando archivo: {error_msg}"}), 400
        
        processed_data = processing_result['processed_data']
        detected_type = processing_result['file_info'].get('detected_type', 'unknown')
        
        if detected_type == 'sql':
            optimized_sql = processed_data
            df = None
            output_ext = ".sql"
        else:
            df = processed_data if isinstance(processed_data, pd.DataFrame) else pd.DataFrame({'data': [str(processed_data)]})
            optimized_sql = None
            output_ext = ".csv"
    except Exception as e:
        if temp_file:
            os.remove(ruta)
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    base_name = os.path.splitext(nombre)[0]
    salida = os.path.join(PROCESSED_FOLDER, f"optimizado_{base_name}{output_ext}")
    
    if df is not None:
        df.to_csv(salida, index=False, na_rep="NaN")
    else:
        with open(salida, 'w', encoding='utf-8') as f:
            f.write(optimized_sql)

    if temp_file:
        os.remove(ruta)

    # Upload to Drive
    try:
        file_metadata = {"name": os.path.basename(salida), "parents": [DRIVE_FOLDER_ID]}
        
        if output_ext == ".sql":
            mimetype = "text/plain"
        else:
            mimetype = "text/csv"
            
        media = MediaFileUpload(salida, mimetype=mimetype)
        uploaded = execute_with_retry(drive_service.files().create(body=file_metadata, media_body=media, fields="id, webViewLink"))

        drive_id = uploaded["id"]
        drive_link = uploaded["webViewLink"]

        execute_with_retry(drive_service.permissions().create(fileId=drive_id, body={"type": "anyone", "role": "reader"}))

    except Exception as e:
        return jsonify({"error": f"No se pudo subir a Google Drive: {e}"}), 500

    return jsonify({"success": True, "ruta_local": salida, "drive_id": drive_id, "drive_link": drive_link})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
