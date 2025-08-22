from flask import Flask, request, send_file, render_template, render_template_string
import os
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



@app.route('/')
def formulario():
    return render_template("index.html")


@app.route('/procesar', methods=['POST'])
def procesar():
    if 'archivo' not in request.files:
        return "No se envió ningún archivo", 400
    archivo = request.files['archivo']
    if archivo.filename == '':
        return "Nombre de archivo inválido", 400
    ruta_entrada = os.path.join(UPLOAD_FOLDER, archivo.filename)
    archivo.save(ruta_entrada)
    extension = os.path.splitext(archivo.filename)[1].lower()
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
            return "Formato no soportado", 400
    except Exception as e:
        return f"Error al procesar: {e}", 500
    salida = os.path.join(PROCESSED_FOLDER, "mejorado_" + archivo.filename + ".csv")
    df.to_csv(salida, index=False, na_rep="NaN")
    return send_file(salida, as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

