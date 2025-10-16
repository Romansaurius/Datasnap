from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json, mysql.connector, requests, time, logging
from parsers.csv_parser import process_csv
from parsers.txt_parser import process_txt
from parsers.xlsx_parser import process_xlsx
from parsers.json_parser import process_json
from parsers.sql_parser import process_sql
from optimizers.universal_data_optimizer import UniversalDataOptimizer
from optimizers.perfect_sql_optimizer import PerfectSQLOptimizer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd

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

# Inicializar optimizadores avanzados
universal_optimizer = UniversalDataOptimizer()
sql_optimizer = PerfectSQLOptimizer()

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

def generate_statistics(df, file_type, optimization_result=None):
    """Genera estadísticas completas del archivo optimizado"""
    stats = {
        "archivo": {
            "tipo": file_type,
            "filas_originales": len(df),
            "columnas": len(df.columns),
            "tamaño_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        },
        "calidad_datos": {
            "valores_nulos": int(df.isnull().sum().sum()),
            "duplicados": int(df.duplicated().sum()),
            "completitud": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        },
        "optimizaciones": {
            "mejoras_aplicadas": 0,
            "problemas_corregidos": 0,
            "confianza": 95.0
        }
    }
    
    # Estadísticas específicas por tipo
    if file_type == "sql" and optimization_result:
        stats["sql"] = {
            "tablas_creadas": optimization_result.tables_created,
            "indices_creados": optimization_result.indexes_created,
            "constraints_agregados": optimization_result.constraints_added,
            "nivel_normalizacion": optimization_result.normalization_level,
            "score_seguridad": round(optimization_result.security_score * 100, 1),
            "score_rendimiento": round(optimization_result.performance_score * 100, 1)
        }
        stats["optimizaciones"]["mejoras_aplicadas"] = len(optimization_result.improvements)
        stats["optimizaciones"]["problemas_corregidos"] = len(optimization_result.issues_found)
        stats["optimizaciones"]["confianza"] = round(optimization_result.confidence_score * 100, 1)
    
    # Análisis de columnas
    stats["columnas_detalle"] = {}
    for col in df.columns:
        col_stats = {
            "tipo": str(df[col].dtype),
            "valores_unicos": int(df[col].nunique()),
            "nulos": int(df[col].isnull().sum()),
            "completitud": round((1 - df[col].isnull().sum() / len(df)) * 100, 1)
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "minimo": float(df[col].min()) if not df[col].empty else 0,
                "maximo": float(df[col].max()) if not df[col].empty else 0,
                "promedio": round(float(df[col].mean()), 2) if not df[col].empty else 0
            })
        
        stats["columnas_detalle"][col] = col_stats
    
    return stats

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

        # Lógica de descarga desde Google Drive (mantenida del original)
        if result.get('drive_id_original'):
            try:
                file_info = drive_service.files().get(fileId=result['drive_id_original'], fields="id,name,parents").execute()
            except Exception as e:
                if "File not found" in str(e) or "notFound" in str(e) or "forbidden" in str(e).lower():
                    if result['ruta'] and os.path.exists(result['ruta']):
                        ruta = result['ruta']
                    else:
                        return jsonify({"error": f"Archivo no accesible. Drive ID: {result['drive_id_original']}, Ruta local: {result['ruta']}"}), 404
                else:
                    return jsonify({"error": f"Error verificando archivo en Google Drive: {e}"}), 500
            else:
                try:
                    drive_request = drive_service.files().get_media(fileId=result['drive_id_original'])
                    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{id_archivo}_{result['nombre']}")
                    with open(temp_path, 'wb') as f:
                        f.write(execute_with_retry(drive_request))
                    ruta = temp_path
                    temp_file = True
                except Exception as e:
                    if result['ruta'] and os.path.exists(result['ruta']):
                        ruta = result['ruta']
                    else:
                        return jsonify({"error": f"No se pudo descargar desde Drive ni encontrar archivo local: {e}"}), 500
        else:
            if not result['ruta'] or not os.path.exists(result['ruta']):
                return jsonify({"error": f"No se encontró el archivo local: {result['ruta']}"}), 404
            ruta = result['ruta']
    except Exception as e:
        return jsonify({"error": f"Error al conectar con la base de datos: {e}"}), 500

    extension = os.path.splitext(ruta)[1].lower()
    logging.info("Procesando archivo con extensión: %s, ruta: %s", extension, ruta)
    
    try:
        optimization_result = None
        
        if extension == ".csv":
            df = process_csv(ruta, HISTORIAL_FOLDER)
            # Aplicar optimización universal
            df, improvements = universal_optimizer.optimize_universal_data(df)
            file_type = "csv"
        elif extension == ".txt":
            df = process_txt(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
            file_type = "txt"
        elif extension == ".xlsx":
            df = process_xlsx(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
            file_type = "xlsx"
        elif extension == ".json":
            df = process_json(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
            file_type = "json"
        elif extension == ".sql":
            # Procesar SQL con optimizador avanzado
            with open(ruta, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            optimization_result = sql_optimizer.optimize_sql(sql_content)
            optimized_sql = optimization_result.optimized_sql
            
            # Crear DataFrame dummy para estadísticas
            df = pd.DataFrame({'optimized': [True]})
            file_type = "sql"
        else:
            return jsonify({"error": "Formato no soportado"}), 400
            
        logging.info("Procesamiento completado")
    except Exception as e:
        logging.error("Error al procesar: %s", e)
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    # Generar estadísticas completas
    stats = generate_statistics(df, file_type, optimization_result)

    if extension == ".sql":
        # Para SQL, guardar el archivo optimizado
        salida = os.path.join(PROCESSED_FOLDER, f"optimizado_{os.path.basename(ruta)}")
        with open(salida, 'w', encoding='utf-8') as f:
            f.write(optimized_sql)
        
        archivo_optimizado = optimized_sql
    else:
        # Para otros formatos, guardar como CSV
        salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{os.path.basename(ruta)}.csv")
        df.to_csv(salida, index=False, na_rep="NaN")
        
        with open(salida, 'r', encoding='utf-8') as f:
            archivo_optimizado = f.read()

    # Limpiar archivo temporal
    if temp_file:
        os.remove(ruta)

    # Actualizar base de datos
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE archivos
            SET estado = 'optimizado'
            WHERE id = %s
        """, (id_archivo,))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error("Error actualizando DB: %s", e)
        return jsonify({"error": f"No se pudo actualizar la base de datos: {e}"}), 500

    return jsonify({
        "success": True,
        "archivo_id": id_archivo,
        "archivo_optimizado": archivo_optimizado,
        "nombre_archivo": f"optimizado_{result['nombre']}",
        "estadisticas": stats,
        "mejoras_aplicadas": optimization_result.improvements if optimization_result else improvements if 'improvements' in locals() else []
    })

@app.route('/debug_file', methods=['POST'])
def debug_file():
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "No se envió el ID del archivo"}), 400

    id_archivo = data['id']
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM archivos WHERE id = %s", (id_archivo,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({"error": "Archivo no encontrado", "id": id_archivo})
            
        debug_info = {
            "archivo_db": result,
            "ruta_existe": os.path.exists(result['ruta']) if result['ruta'] else False,
            "drive_id_original": result.get('drive_id_original'),
            "working_directory": os.getcwd(),
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER),
            "upload_folder_contents": os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else []
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": f"Error en debug: {e}"})

@app.route('/procesar_drive', methods=['POST'])
def procesar_drive():
    data = request.json
    if not data or 'drive_file_id' not in data:
        return jsonify({"error": "No se envió el ID del archivo de Google Drive"}), 400

    drive_file_id = data['drive_file_id']

    try:
        # Download from Drive
        drive_request = drive_service.files().get_media(fileId=drive_file_id)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_drive_{drive_file_id}")
        with open(temp_path, 'wb') as f:
            f.write(execute_with_retry(drive_request))

        # Get file name
        file_info = execute_with_retry(drive_service.files().get(fileId=drive_file_id, fields="name"))
        nombre = file_info['name']
        ruta = temp_path
        temp_file = True
    except Exception as e:
        return jsonify({"error": f"Error al descargar desde Google Drive: {e}"}), 500

    extension = os.path.splitext(nombre)[1].lower()
    try:
        if extension == ".csv":
            df = process_csv(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
        elif extension == ".txt":
            df = process_txt(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
        elif extension == ".xlsx":
            df = process_xlsx(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
        elif extension == ".json":
            df = process_json(ruta, HISTORIAL_FOLDER)
            df, improvements = universal_optimizer.optimize_universal_data(df)
        elif extension == ".sql":
            with open(ruta, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            optimization_result = sql_optimizer.optimize_sql(sql_content)
            df = pd.DataFrame({'optimized': [True]})
        else:
            if temp_file:
                os.remove(ruta)
            return jsonify({"error": "Formato no soportado"}), 400
    except Exception as e:
        if temp_file:
            os.remove(ruta)
        return jsonify({"error": f"Error al procesar: {e}"}), 500

    if extension == ".sql":
        salida = os.path.join(PROCESSED_FOLDER, f"optimizado_{nombre}")
        with open(salida, 'w', encoding='utf-8') as f:
            f.write(optimization_result.optimized_sql)
    else:
        salida = os.path.join(PROCESSED_FOLDER, f"mejorado_{nombre}")
        df.to_csv(salida, index=False, na_rep="NaN")

    if temp_file:
        os.remove(ruta)

    # Upload to Drive
    try:
        file_metadata = {"name": os.path.basename(salida), "parents": [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(salida, mimetype="text/csv" if extension != ".sql" else "text/plain")
        uploaded = execute_with_retry(drive_service.files().create(body=file_metadata, media_body=media, fields="id, webViewLink"))

        drive_id = uploaded["id"]
        drive_link = uploaded["webViewLink"]

        execute_with_retry(drive_service.permissions().create(fileId=drive_id, body={"type": "anyone", "role": "reader"}))

    except Exception as e:
        return jsonify({"error": f"No se pudo subir a Google Drive: {e}"}), 500

    # Generar estadísticas
    stats = generate_statistics(df, extension[1:], optimization_result if extension == ".sql" else None)

    return jsonify({
        "success": True, 
        "ruta_local": salida, 
        "drive_id": drive_id, 
        "drive_link": drive_link,
        "estadisticas": stats
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)